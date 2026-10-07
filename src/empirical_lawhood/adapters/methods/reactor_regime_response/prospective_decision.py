"""Outcome-blind four-request and six-policy D decision seal."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar, cast

from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .calibration_pipeline import RegimeCalibrationPackage
from .config import ROOTS
from .causal_contexts import causal_contexts
from .comparison_math import POLICIES, fixed_choice, point_choice
from .control_math import REQUESTS_K, WordForecast, forecast_chart, select_word
from .mechanistic_comparator import mechanistic_chart
from .model_records import RegimeFitPackage
from .nomination_records import RegimeNominationPackage
from .prediction_seal import seal_prospective_predictions
from .causal_preparation import causal_preparation_validity
from .qualification_pipeline import RegimeQualificationPackage
from .law_terminal import RegimeJointLawResult, RESULT_ID
from .release import require_scientific_release
from .records import RegimeCausalPreparation, RegimePredictionSeal

FourChoices = tuple[int | None, int | None, int | None, int | None]


def _four(values: tuple[int | None, ...]) -> FourChoices:
    if len(values) != 4:
        raise ValueError("consumer changed its four cooling requests")
    return values


@dataclass(frozen=True, slots=True)
class ProspectiveWordForecast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/prospective-word-forecast'

    word: int
    requested_feed_kg_s: D
    delivered_mass_kg: D
    predicted_peak_K: D
    predicted_cooling_K: D
    temperature_halfwidth_K: D
    cooling_halfwidth_K: D
    supported: bool
    projection_valid: bool

    @classmethod
    def from_word(cls, value: WordForecast) -> ProspectiveWordForecast:
        return cls(
            value.word,
            D(repr(value.requested_feed_kg_s)),
            D(repr(value.delivered_mass_kg)),
            D(repr(value.predicted_peak_K)),
            D(repr(value.predicted_cooling_K)),
            D(repr(value.temperature_halfwidth_K)),
            D(repr(value.cooling_halfwidth_K)),
            value.supported,
            value.projection_valid,
        )

    def to_word(self) -> WordForecast:
        return WordForecast(
            self.word,
            float(self.requested_feed_kg_s),
            float(self.delivered_mass_kg),
            float(self.predicted_peak_K),
            float(self.predicted_cooling_K),
            float(self.temperature_halfwidth_K),
            float(self.cooling_halfwidth_K),
            self.supported,
            self.projection_valid,
        )


@dataclass(frozen=True, slots=True)
class ProspectivePolicyChoices(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/prospective-policy-choices'

    policy: str
    choices: tuple[int | None, int | None, int | None, int | None]
    evaluable: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.policy not in POLICIES
            or len(self.choices) != 4
            or any(value not in (None, 1, 2) for value in self.choices)
            or (not self.evaluable and not self.reasons)
        ):
            raise ValueError("D policy seal changed its four-request shadow choice census")


@dataclass(frozen=True, slots=True)
class TruthScreenedRegimeSoftwareDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reactor-regime/truth-screened-regime-software-decision'

    decision_id: str
    root: str
    route: str
    callback: int
    causal_preparation: ObjectIdentity
    private_preparation: ObjectIdentity
    prediction_seal: ObjectIdentity
    fit_package: ObjectIdentity
    nomination: ObjectIdentity
    calibration: ObjectIdentity
    qualification: ObjectIdentity
    qualified_law: ObjectIdentity
    issue_authority: ObjectIdentity
    preparation_safe: bool
    preparation_maxima_K: tuple[D | None, D | None]
    preparation_reasons: tuple[str, ...]
    primary_words: tuple[ProspectiveWordForecast, ...]
    direct_words: tuple[ProspectiveWordForecast, ...]
    mechanistic_words: tuple[ProspectiveWordForecast, ...]
    policies: tuple[ProspectivePolicyChoices, ...]
    refusal_reasons: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if (
            self.decision_id != f"{self.root}.four-request-decision"
            or self.route not in ("prepared_t0", "c_q", "p_q")
            or not 0 < self.callback < 2880
            or self.causal_preparation.object_id != f"{self.root}.causal-preparation"
            or self.private_preparation.object_id != f"{self.root}.private-preparation"
            or self.prediction_seal.object_id != f"{self.root}.prediction-seal"
            or tuple(value.word for value in self.primary_words) != (0, 1, 2)
            or self.direct_words and tuple(value.word for value in self.direct_words) != (0, 1, 2)
            or self.mechanistic_words and tuple(value.word for value in self.mechanistic_words) != (0, 1, 2)
            or tuple(value.policy for value in self.policies) != POLICIES
            or len(self.refusal_reasons) != 4
            or len(self.preparation_maxima_K) != 2
            or (not self.preparation_safe and not self.preparation_reasons)
            or (self.preparation_safe and (
                bool(self.preparation_reasons)
                or any(value is None or value > D("356.2") for value in self.preparation_maxima_K)
            ))
        ):
            raise ValueError("D decision changed its complete sealed local consumer")


@dataclass(frozen=True, slots=True)
class TruthScreenedRegimeSoftwareAssignment(CanonicalRecord):
    """One D root keeps its assigned place even when no decision is possible."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reactor-regime/truth-screened-regime-software-assignment'

    assignment_id: str
    root: str
    seed: int
    causal_preparation: ObjectIdentity
    prediction_seal: ObjectIdentity
    qualified_law: ObjectIdentity
    issue_authority: ObjectIdentity
    decision: TruthScreenedRegimeSoftwareDecision | None
    nonentry_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.assignment_id != f"{self.root}.prospective-assignment"
            or (self.root, self.seed)
            not in {(root, seed) for root, role, _, seed in ROOTS if role == "prospective"}
            or self.causal_preparation.object_id != f"{self.root}.causal-preparation"
            or self.prediction_seal.object_id != f"{self.root}.prediction-seal"
            or (self.decision is None) != bool(self.nonentry_reasons)
            or (self.decision is not None and (
                self.decision.root != self.root
                or self.decision.causal_preparation != self.causal_preparation
                or self.decision.prediction_seal != self.prediction_seal
                or self.decision.qualified_law != self.qualified_law
                or self.decision.issue_authority != self.issue_authority
            ))
        ):
            raise ValueError("D assignment loses its immutable root/decision census")


# Truth-screened decisions remain decode-only precontact software evidence.
# Causal-validity decisions must be derived afresh from permitted causal inputs;
# truth-gated choices cannot migrate.
@dataclass(frozen=True, slots=True)
class CausalValidityRegimeDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reactor-regime/causal-validity-regime-decision'

    decision_id: str
    root: str
    route: str
    callback: int
    causal_preparation: ObjectIdentity
    prediction_seal: ObjectIdentity
    fit_package: ObjectIdentity
    nomination: ObjectIdentity
    calibration: ObjectIdentity
    qualification: ObjectIdentity
    qualified_law: ObjectIdentity
    issue_authority: ObjectIdentity
    causal_preparation_valid: bool
    causal_reasons: tuple[str, ...]
    primary_words: tuple[ProspectiveWordForecast, ...]
    direct_words: tuple[ProspectiveWordForecast, ...]
    mechanistic_words: tuple[ProspectiveWordForecast, ...]
    policies: tuple[ProspectivePolicyChoices, ...]
    refusal_reasons: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if (
            self.decision_id != f"{self.root}.four-request-decision"
            or self.route not in ("prepared_t0", "c_q", "p_q")
            or not 0 < self.callback < 2880
            or self.causal_preparation.object_id != f"{self.root}.causal-preparation"
            or self.prediction_seal.object_id != f"{self.root}.prediction-seal"
            or tuple(value.word for value in self.primary_words) != (0, 1, 2)
            or self.direct_words and tuple(value.word for value in self.direct_words) != (0, 1, 2)
            or self.mechanistic_words and tuple(value.word for value in self.mechanistic_words) != (0, 1, 2)
            or tuple(value.policy for value in self.policies) != POLICIES
            or len(self.refusal_reasons) != 4
            or self.causal_preparation_valid == bool(self.causal_reasons)
        ):
            raise ValueError("D decision changed its complete sealed local consumer")


@dataclass(frozen=True, slots=True)
class CausalValidityRegimeAssignment(CanonicalRecord):
    """One D root keeps its assigned place even when no decision is possible."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reactor-regime/causal-validity-regime-assignment'

    assignment_id: str
    root: str
    seed: int
    causal_preparation: ObjectIdentity
    prediction_seal: ObjectIdentity
    qualified_law: ObjectIdentity
    issue_authority: ObjectIdentity
    decision: CausalValidityRegimeDecision | None
    nonentry_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.assignment_id != f"{self.root}.prospective-assignment"
            or (self.root, self.seed)
            not in {(root, seed) for root, role, _, seed in ROOTS if role == "prospective"}
            or self.causal_preparation.object_id != f"{self.root}.causal-preparation"
            or self.prediction_seal.object_id != f"{self.root}.prediction-seal"
            or (self.decision is None) != bool(self.nonentry_reasons)
            or (self.decision is not None and (
                self.decision.root != self.root
                or self.decision.causal_preparation != self.causal_preparation
                or self.decision.prediction_seal != self.prediction_seal
                or self.decision.qualified_law != self.qualified_law
                or self.decision.issue_authority != self.issue_authority
            ))
        ):
            raise ValueError("D assignment loses its immutable root/decision census")


def _model_forecast(
    seal: RegimePredictionSeal,
    context_index: int,
    coefficient_id: str,
    absolute_id: str,
    q_t: float,
    q_c: float,
) -> tuple[WordForecast, WordForecast, WordForecast] | None:
    arrays = seal.predictions.unpack()
    ids = {model_id: index for index, model_id in enumerate(seal.candidate_ids)}
    keys = (f"coefficient.{coefficient_id}".lower(), f"absolute.{absolute_id}".lower())
    if any(key not in ids for key in keys):
        raise ValueError("D prediction seal lacks a frozen selected model")
    k, absolute_prediction_index = (ids[key] for key in keys)
    if not (arrays["present"][k, context_index] and arrays["present"][absolute_prediction_index, context_index]):
        return None
    masses = tuple(float(value) for value in arrays["masses"][context_index])
    if not masses[0] == 0 < masses[1] < masses[2]:
        return None
    supported = bool(arrays["support"][k, context_index] and arrays["support"][absolute_prediction_index, context_index])
    return forecast_chart(
        float(arrays["forecast"][absolute_prediction_index, context_index]),
        float(arrays["forecast"][k, context_index]),
        masses,  # type: ignore[arg-type]
        q_t,
        q_c,
        (supported,) * 3,
        tuple(bool(value) for value in arrays["projection_valid"][context_index]),  # type: ignore[arg-type]
    )  # type: ignore[return-value]


def seal_prospective_decision(
    source: ReactorBatchSource,
    causal: RegimeCausalPreparation,
    seal: RegimePredictionSeal,
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
    calibration: RegimeCalibrationPackage,
    qualification: RegimeQualificationPackage,
    law: RegimeJointLawResult,
    issue_authority: ObjectIdentity,
) -> CausalValidityRegimeDecision:
    """Commit every policy before this root's action outcomes are acquired."""
    require_scientific_release(nomination, qualification, law)
    if (
        causal.role != "prospective"
        or calibration.q_temperature is None
        or calibration.q_cooling is None
        or calibration.fit_package != ObjectIdentity.from_record(fit.package_id, fit)
        or calibration.nomination != ObjectIdentity.from_record(nomination.package_id, nomination)
        or qualification.calibration
        != ObjectIdentity.from_record(calibration.package_id, calibration)
        or seal != seal_prospective_predictions(causal, fit, nomination, qualification, law)
        or source.fingerprint() != causal.source_sha256
    ):
        raise ValueError("D decision lacks its exact qualified source/causal ancestry")
    route = nomination.selected_route
    if route is None:
        raise ValueError("D has no frozen selected preparation")
    context = next(value for value in causal_contexts(causal) if value.name == route)
    causal_valid, causal_reasons = causal_preparation_validity(causal, route)
    if context.callback is None or context.input_sha256 is None:
        raise ValueError("D selected preparation has no causal decision callback")
    chosen = next(value for value in nomination.routes if value.route == route)
    if chosen.coefficient_model_id is None or chosen.absolute_model_id is None:
        raise ValueError("D selected route lacks its frozen primary law")
    context_index = ("early", "middle", "late", "prepared_t0", "c_q", "c_q600", "p_q", "p_q600").index(route)
    primary = _model_forecast(
        seal, context_index, chosen.coefficient_model_id, chosen.absolute_model_id,
        float(calibration.q_temperature), float(calibration.q_cooling),
    )
    if primary is None:
        raise ValueError("D selected law cannot forecast its assigned causal context")
    direct = (
        None if chosen.direct_model_id is None else _model_forecast(
            seal, context_index, chosen.direct_model_id, chosen.absolute_model_id, 0, 0,
        )
    )
    mech = mechanistic_chart(source, causal, route)
    mech_words: tuple[WordForecast, ...] = ()
    if mech is not None:
        masses = tuple(word.delivered_mass_kg for word in primary)
        mech_words = tuple(
            WordForecast(
                word, (0.0, 0.016, 0.032)[word], masses[word],
                mech[0][word], mech[1][word], 0, 0,
                primary[word].projection_valid, primary[word].projection_valid,
            )
            for word in range(3)
        )
    el_admissions = tuple(
        select_word(
            request,
            primary,
            law_valid=True,
            preparation_safe=causal_valid,
            clock_valid=True,
            authority_valid=True,
            numerical_valid=True,
        )
        for request in REQUESTS_K
    )
    choices = (
        ProspectivePolicyChoices("EL", _four(tuple(item.choice for item in el_admissions)), True, ()),
        ProspectivePolicyChoices(
            "POINT", _four(tuple(point_choice(request, primary) for request in REQUESTS_K)), True, (),
        ),
        ProspectivePolicyChoices(
            "DIRECT",
            (None,) * 4 if direct is None else _four(tuple(point_choice(request, direct) for request in REQUESTS_K)),
            direct is not None,
            () if direct is not None else ("DIRECT_FIT_OR_CONTEXT_UNAVAILABLE",),
        ),
        ProspectivePolicyChoices(
            "MECH",
            (None,) * 4 if not mech_words else _four(tuple(point_choice(request, cast(tuple[WordForecast, WordForecast, WordForecast], mech_words)) for request in REQUESTS_K)),
            bool(mech_words),
            () if mech_words else ("NOMINAL_MECHANISTIC_PREFIX_UNAVAILABLE",),
        ),
        ProspectivePolicyChoices(
            "FIXED_FEED_SIXTEEN_GRAMS_PER_SECOND", (fixed_choice(1, primary[1].projection_valid),) * 4, True, (),
        ),
        ProspectivePolicyChoices(
            "FIXED_FEED_THIRTY_TWO_GRAMS_PER_SECOND", (fixed_choice(2, primary[2].projection_valid),) * 4, True, (),
        ),
    )
    return CausalValidityRegimeDecision(
        f"{causal.root}.four-request-decision",
        causal.root,
        route,
        context.callback,
        ObjectIdentity.from_record(f"{causal.root}.causal-preparation", causal),
        ObjectIdentity.from_record(f"{causal.root}.prediction-seal", seal),
        ObjectIdentity.from_record(fit.package_id, fit),
        ObjectIdentity.from_record(nomination.package_id, nomination),
        ObjectIdentity.from_record(calibration.package_id, calibration),
        ObjectIdentity.from_record(qualification.package_id, qualification),
        ObjectIdentity.from_record(RESULT_ID, law),
        issue_authority,
        causal_valid,
        causal_reasons,
        tuple(ProspectiveWordForecast.from_word(value) for value in primary),
        () if direct is None else tuple(ProspectiveWordForecast.from_word(value) for value in direct),
        tuple(ProspectiveWordForecast.from_word(value) for value in mech_words),
        choices,
        tuple(item.reasons for item in el_admissions),
    )


def seal_prospective_assignment(
    source: ReactorBatchSource,
    causal: RegimeCausalPreparation,
    seal: RegimePredictionSeal,
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
    calibration: RegimeCalibrationPackage,
    qualification: RegimeQualificationPackage,
    law: RegimeJointLawResult,
    issue_authority: ObjectIdentity,
) -> CausalValidityRegimeAssignment:
    """Seal an actual choice or an exact no-contact/nonforecast root nonentry."""
    require_scientific_release(nomination, qualification, law)
    if (
        causal.role != "prospective"
        or source.fingerprint() != causal.source_sha256
        or seal != seal_prospective_predictions(causal, fit, nomination, qualification, law)
    ):
        raise ValueError("D assignment changed its released source, prefix or prediction seal")
    route = nomination.selected_route
    if route is None:
        raise ValueError("released D assignment has no selected route")
    context = next(value for value in causal_contexts(causal) if value.name == route)
    reasons: tuple[str, ...] = ()
    if context.callback is None or context.input_sha256 is None:
        reasons = ("SELECTED_PREPARATION_NO_CONTACT",)
    else:
        selected = next(value for value in nomination.routes if value.route == route)
        if selected.coefficient_model_id is None or selected.absolute_model_id is None:
            raise ValueError("released D route lacks its selected frozen joint models")
        context_index = (
            "early", "middle", "late", "prepared_t0", "c_q", "c_q600", "p_q", "p_q600"
        ).index(route)
        if _model_forecast(
            seal, context_index, selected.coefficient_model_id,
            selected.absolute_model_id, 0, 0,
        ) is None:
            reasons = ("SELECTED_CAUSAL_FEATURE_OR_PROJECTION_UNAVAILABLE",)
    decision = None if reasons else seal_prospective_decision(
        source, causal, seal, fit, nomination, calibration,
        qualification, law, issue_authority,
    )
    return CausalValidityRegimeAssignment(
        f"{causal.root}.prospective-assignment",
        causal.root,
        causal.seed,
        ObjectIdentity.from_record(f"{causal.root}.causal-preparation", causal),
        ObjectIdentity.from_record(f"{causal.root}.prediction-seal", seal),
        ObjectIdentity.from_record(RESULT_ID, law),
        issue_authority,
        decision,
        reasons,
    )
