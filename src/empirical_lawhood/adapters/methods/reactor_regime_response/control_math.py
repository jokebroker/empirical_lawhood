"Frozen local reactor prediction, admission and paired-receiver arithmetic.\n\nThis is an experiment operand, not a replacement for the installed admission/controller use\nprogramme, owner delivery, custody or immutable adjudication.\n"

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from scipy.stats import beta

REQUESTS_K = (0.0004, 0.0008, 0.0012, 0.0016)
WORDS_KG_S = (0.0, 0.016, 0.032)
TEMPERATURE_LIMIT_K = 356.2


@dataclass(frozen=True)
class WordForecast:
    word: int
    requested_feed_kg_s: float
    delivered_mass_kg: float
    predicted_peak_K: float
    predicted_cooling_K: float
    temperature_halfwidth_K: float
    cooling_halfwidth_K: float
    supported: bool
    projection_valid: bool

    def __post_init__(self) -> None:
        if (
            self.word not in (0, 1, 2)
            or self.requested_feed_kg_s != WORDS_KG_S[self.word]
            or not all(
                isfinite(x)
                for x in (
                    self.delivered_mass_kg,
                    self.predicted_peak_K,
                    self.predicted_cooling_K,
                    self.temperature_halfwidth_K,
                    self.cooling_halfwidth_K,
                )
            )
            or self.delivered_mass_kg < 0
            or self.temperature_halfwidth_K < 0
            or self.cooling_halfwidth_K < 0
        ):
            raise ValueError("invalid native word forecast")
        if self.word == 0 and (
            self.delivered_mass_kg != 0 or self.predicted_cooling_K != 0
        ):
            raise ValueError("zero-feed reference must remain the zero response")

    @property
    def cooling_lower_K(self) -> float:
        return self.predicted_cooling_K - self.cooling_halfwidth_K

    @property
    def temperature_upper_K(self) -> float:
        return self.predicted_peak_K + self.temperature_halfwidth_K


@dataclass(frozen=True)
class Admission:
    request_K: float
    choice: int | None
    admitted_words: tuple[int, ...]
    reasons: tuple[str, ...]

    @property
    def act(self) -> bool:
        return self.choice is not None


def forecast_chart(
    p0_hat_K: float,
    kappa_hat_K_per_kg: float,
    masses_kg: tuple[float, float, float],
    q_temperature: float,
    q_cooling: float,
    support: tuple[bool, bool, bool],
    projection_valid: tuple[bool, bool, bool],
) -> tuple[WordForecast, ...]:
    if (
        not all(isfinite(x) for x in (p0_hat_K, kappa_hat_K_per_kg, q_temperature, q_cooling))
        or q_temperature < 0
        or q_cooling < 0
        or len(masses_kg) != 3
        or len(support) != 3
        or len(projection_valid) != 3
        or masses_kg[0] != 0
        or not 0 < masses_kg[1] < masses_kg[2]
    ):
        raise ValueError("forecast lacks a finite distinct native chart")
    values = []
    for word, mass in enumerate(masses_kg):
        cooling = kappa_hat_K_per_kg * mass
        # An out-of-chart prediction is unsupported, never clipped.
        word_supported = support[word] and abs(cooling) <= 0.01
        values.append(
            WordForecast(
                word,
                WORDS_KG_S[word],
                mass,
                p0_hat_K - cooling,
                cooling,
                q_temperature * 0.25 + 0.01,
                q_cooling * (0.00005 + 0.5 * abs(cooling)) + 0.000001,
                word_supported,
                projection_valid[word],
            )
        )
    return tuple(values)


def select_word(
    request_K: float,
    chart: tuple[WordForecast, ...],
    *,
    law_valid: bool,
    preparation_safe: bool,
    clock_valid: bool,
    authority_valid: bool,
    numerical_valid: bool,
) -> Admission:
    if request_K not in REQUESTS_K or tuple(w.word for w in chart) != (0, 1, 2):
        raise ValueError("consumer request or complete native chart differs")
    common = {
        "LAW_INVALID": law_valid,
        "PREPARATION_UNSAFE": preparation_safe,
        "CLOCK_INVALID": clock_valid,
        "AUTHORITY_INVALID": authority_valid,
        "NUMERICAL_INVALID": numerical_valid,
    }
    blockers = {reason for reason, allowed in common.items() if not allowed}
    admitted = []
    for word in chart[1:]:
        if not word.supported:
            blockers.add("OUTSIDE_SUPPORT")
        elif not word.projection_valid:
            blockers.add("INVALID_NATIVE_PROJECTION")
        elif word.cooling_lower_K < request_K:
            blockers.add("COOLING_LOWER_BELOW_REQUEST")
        elif word.temperature_upper_K > TEMPERATURE_LIMIT_K:
            blockers.add("TEMPERATURE_UPPER_ABOVE_LIMIT")
        else:
            admitted.append(word)
    if any(not valid for valid in common.values()):
        admitted.clear()
    selected = min(
        admitted,
        key=lambda word: (word.delivered_mass_kg, word.requested_feed_kg_s, word.word),
        default=None,
    )
    return Admission(
        request_K,
        None if selected is None else selected.word,
        tuple(word.word for word in admitted),
        tuple(sorted(blockers)) if selected is None else (),
    )


@dataclass(frozen=True)
class MeasuredWord:
    word: int
    view: int
    measured_peak_K: float
    measured_cooling_K: float
    grid_temperature_K: tuple[float, ...]
    delivered_mass_kg: float
    dose_valid: bool
    window_valid: bool
    receipt_valid: bool
    observation_valid: bool

    def __post_init__(self) -> None:
        if (
            self.word not in (0, 1, 2)
            or self.view not in (0, 1)
            or not self.grid_temperature_K
            or not all(isfinite(v) for v in self.grid_temperature_K)
            or not all(
                isfinite(v)
                for v in (self.measured_peak_K, self.measured_cooling_K, self.delivered_mass_kg)
            )
            or self.measured_peak_K != max(self.grid_temperature_K)
        ):
            raise ValueError("paired receiver must retain its complete finite native window")


def covered(word: WordForecast, outcome: MeasuredWord) -> bool:
    return (
        word.word == outcome.word
        and outcome.receipt_valid
        and outcome.observation_valid
        and outcome.dose_valid
        and outcome.window_valid
        and abs(word.predicted_peak_K - outcome.measured_peak_K)
        <= word.temperature_halfwidth_K
        and abs(word.predicted_cooling_K - outcome.measured_cooling_K)
        <= word.cooling_halfwidth_K
    )


@dataclass(frozen=True)
class RootControllerUseEvaluation:
    adequate: bool
    successes: tuple[bool, bool, bool, bool]
    joined: bool
    false_admission: bool
    evaluability_reasons: tuple[str, ...]


def evaluate_root(
    chart: tuple[WordForecast, ...],
    measured: tuple[MeasuredWord, ...],
    decisions: tuple[Admission, ...],
    *,
    preparation_safe: bool,
    numerically_valid: bool,
    delivered_words: tuple[int | None, ...],
    primary_outcomes: tuple[tuple[MeasuredWord, MeasuredWord] | None, ...],
) -> RootControllerUseEvaluation:
    if (
        tuple(w.word for w in chart) != (0, 1, 2)
        or tuple(d.request_K for d in decisions) != REQUESTS_K
        or len(delivered_words) != 4
        or len(primary_outcomes) != 4
    ):
        raise ValueError("Controller-use root lost its complete chart or four requests")
    outcomes = {(m.word, m.view): m for m in measured}
    complete = len(outcomes) == 6 and len(measured) == 6
    reasons = set()
    if not complete:
        reasons.add("MISSING_COMPLETE_PAIRED_CHART")
    if not numerically_valid:
        reasons.add("NUMERICAL_INVALID")
    if not preparation_safe:
        reasons.add("PREPARATION_UNSAFE")
    adequate = (
        complete
        and preparation_safe
        and numerically_valid
        and all(w.supported and w.projection_valid for w in chart)
        and all(covered(w, outcomes[(w.word, view)]) for w in chart for view in (0, 1))
    )
    successes = []
    false_admission = False
    selected_deliveries_adequate = True
    for decision, delivered, primary in zip(
        decisions, delivered_words, primary_outcomes, strict=True
    ):
        chosen = decision.choice
        if chosen is None:
            successes.append(False)
            continue
        selected = chart[chosen]
        physical_success = False
        selected_covered = False
        if primary is not None:
            physical_success = (
                tuple(view.view for view in primary) == (0, 1)
                and all(view.word == chosen for view in primary)
                and delivered == chosen
                and preparation_safe
                and numerically_valid
                and selected.supported
                and selected.projection_valid
                and all(
                    primary[view].receipt_valid
                    and primary[view].observation_valid
                    and primary[view].dose_valid
                    and primary[view].window_valid
                    and primary[view].measured_cooling_K >= decision.request_K
                    and max(primary[view].grid_temperature_K) <= TEMPERATURE_LIMIT_K
                    for view in (0, 1)
                )
            )
            selected_covered = all(covered(selected, view) for view in primary)
        selected_deliveries_adequate = (
            selected_deliveries_adequate
            and delivered == chosen
            and selected.supported
            and selected.projection_valid
            and selected_covered
        )
        successes.append(bool(physical_success))
        if not physical_success or not selected_covered:
            false_admission = True
            reasons.add("ADMITTED_DELIVERY_UNSUPPORTED_OR_UNVALIDATED")
    success_tuple = tuple(successes)
    adequate = bool(adequate and selected_deliveries_adequate)
    return RootControllerUseEvaluation(
        adequate,
        success_tuple,  # type: ignore[arg-type]
        bool(adequate and all(success_tuple)),
        false_admission,
        tuple(sorted(reasons)),
    )


def one_sided_cp(successes: int, assignments: int, *, lower: bool) -> float:
    if not 0 <= successes <= assignments or assignments <= 0:
        raise ValueError("invalid independent-root binomial count")
    if lower:
        return 0.0 if successes == 0 else float(beta.ppf(0.05, successes, assignments - successes + 1))
    return 1.0 if successes == assignments else float(
        beta.ppf(0.95, successes + 1, assignments - successes)
    )


def prospective_cohort(roots: tuple[RootControllerUseEvaluation, ...]) -> dict[str, object]:
    if len(roots) != 64:
        raise ValueError("Controller use requires all 64 assigned independent roots")
    joined = sum(r.joined for r in roots)
    false_admission = sum(r.false_admission for r in roots)
    lower = one_sided_cp(joined, 64, lower=True)
    upper = one_sided_cp(false_admission, 64, lower=False)
    return {
        "assignments": 64,
        "joined_successes": joined,
        "false_admission_roots": false_admission,
        "joined_lower_95": lower,
        "false_admission_upper_95": upper,
        "adequate_use": lower >= 0.70 and upper <= 0.10,
    }
