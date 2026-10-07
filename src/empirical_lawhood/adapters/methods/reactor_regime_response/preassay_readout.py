"""Causal comparator forecasts and qualification choices, sealed before assays.

The older shared operator is evaluated only on the older local chart. These
are frozen descriptive comparators, never additional evidence for the new law.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal as D
from hashlib import sha256
from typing import ClassVar, cast

import numpy as np

from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import CausalFeatures, Observation
from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.records import FeedDomain
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256

from .calibration_pipeline import RegimeCalibrationPackage
from .causal_contexts import causal_contexts
from .causal_preparation import causal_preparation_validity
from .config import ROOTS
from .control_math import REQUESTS_K, TEMPERATURE_LIMIT_K, select_word
from .measured_panel import CONTEXTS, measured_contexts
from .mechanistic_comparator import mechanistic_chart
from .nomination_records import RegimeNominationPackage
from .preparation_evidence import preparation_evidence
from .prospective_decision import ProspectiveWordForecast, _model_forecast
from .records import RegimeAssayPanel, RegimeCausalPreparation, RegimePredictionSeal, RegimePrivatePreparation

COMPARATORS = ("OLD_LOCAL_PAIRED", "OLD_SHARED_PAIRED", "MECH")


@dataclass(frozen=True, slots=True)
class RegimeReferenceComparators(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/reactor-regime-response/regime-reference-comparators'
    )
    VERSION: ClassVar[str] = '1.0.0'
    local: FeedDomain
    shared_publication: str
    local_sha256: str
    shared_sha256: str

    def __post_init__(self) -> None:
        validate_sha256(self.local_sha256)
        validate_sha256(self.shared_sha256)
        if (
            self.local.fingerprint() != self.local_sha256
            or sha256(self.shared_publication.encode("utf-8")).hexdigest()
            != self.shared_sha256
        ):
            raise ValueError("retained comparator publication changed")
        value = json.loads(self.shared_publication)
        if len(value["model"]["domains"]) != 1:
            raise ValueError("shared comparator changed its single frozen operator")


def _numbers(values: object) -> tuple[D, ...]:
    return tuple(D(repr(float(v))) for v in np.asarray(values).reshape(-1))


@dataclass(frozen=True, slots=True)
class RegimeComparatorForecast(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/reactor-regime-response/regime-comparator-forecast'
    )
    context: str
    comparator: str
    peak_K: tuple[D, ...]
    cooling_K: tuple[D, ...]
    supported: tuple[bool, bool, bool]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.context not in CONTEXTS
            or self.comparator not in COMPARATORS
            or len(self.peak_K) != len(self.cooling_K)
            or len(self.peak_K) not in (0, 3)
            or any(not value.is_finite() for value in (*self.peak_K, *self.cooling_K))
            or (not self.peak_K and not self.reasons)
        ):
            raise ValueError("comparator changed its causal three-word forecast")


def comparator_forecasts(
    source: ReactorBatchSource,
    references: RegimeReferenceComparators,
    causal: RegimeCausalPreparation,
) -> tuple[RegimeComparatorForecast, ...]:
    if source.fingerprint() != causal.source_sha256:
        raise ValueError("comparator source differs from causal prefix")
    shared = json.loads(references.shared_publication)["model"]["domains"][0]["fit"]
    arrays = causal.arrays.unpack()
    rows = []
    for context in causal_contexts(causal):
        forecasts: dict[str, tuple[object, object, tuple[bool, bool, bool]]] = {}
        if context.input_sha256 is not None and context.callback is not None:
            stem = "exploration_unshifted_v0" if context.name in CONTEXTS[:4] else f"{context.name[0]}_v0"
            observations = arrays[f"{stem}_observations"]
            stages = arrays[f"{stem}_stages"]
            state = CausalFeatures()
            for row in observations[: context.callback + 1]:
                state.append(Observation(*map(float, row)))
            previous = (
                float(stages[context.callback - 1, 2]),
                float(stages[context.callback - 1, 3]),
            )
            observation = Observation(*map(float, observations[context.callback]))
            projections = tuple(
                Actuator().project_request(
                    observation, previous, (feed, previous[1]), 1 + 3 * i
                )
                for i, feed in enumerate((0.0, 0.016, 0.032))
            )
            x = np.asarray([state.features(previous[1], p) for p in projections])
            old = references.local.predict(x)
            support = cast(
                tuple[bool, bool, bool],
                tuple(
                    bool(v)
                    for v in references.local.proposed_support(x, np.asarray((1, 4, 7)))
                ),
            )
            forecasts["OLD_LOCAL_PAIRED"] = (old[:, 0], old[:, 1], support)
            z = x.copy()
            z[:, (5, 11, 21)] = 0
            operator = np.asarray(shared["operator"], dtype=float)
            peaks = ((x - shared["mean"]) / shared["scale"]) @ operator[:-1] + operator[
                -1
            ]
            zero = ((z - shared["mean"]) / shared["scale"]) @ operator[:-1] + operator[
                -1
            ]
            forecasts["OLD_SHARED_PAIRED"] = (
                peaks[:, 0],
                zero[:, 0] - peaks[:, 0],
                support,
            )
            mech = mechanistic_chart(source, causal, context.name)
            if mech is not None:
                forecasts["MECH"] = (*mech, context.projection_valid)
        for name in COMPARATORS:
            forecast = forecasts.get(name)
            rows.append(
                RegimeComparatorForecast(
                    context.name,
                    name,
                    () if forecast is None else _numbers(forecast[0]),
                    () if forecast is None else _numbers(forecast[1]),
                    (False,) * 3 if forecast is None else forecast[2],
                    ("CAUSAL_FORECAST_UNAVAILABLE",) if forecast is None else (),
                )
            )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class RegimePreassayReadout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-preassay-readout'
    root: str
    role: str
    prediction_seal: ObjectIdentity
    references: ObjectIdentity
    comparators: tuple[RegimeComparatorForecast, ...]
    calibration: ObjectIdentity | None
    selected_route: str | None
    causal_preparation_valid: bool
    primary_words: tuple[ProspectiveWordForecast, ...]
    choices: tuple[int | None, ...]
    refusal_reasons: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if (
            (self.root, self.role) not in {(root, role) for root, role, _, _ in ROOTS}
            or self.prediction_seal.object_id != f"{self.root}.prediction-seal"
            or tuple((v.context, v.comparator) for v in self.comparators)
            != tuple((context, name) for context in CONTEXTS for name in COMPARATORS)
            or len(self.choices) != (4 if self.role == "qualification" else 0)
            or len(self.choices) != len(self.refusal_reasons)
            or any(v not in (None, 1, 2) for v in self.choices)
        ):
            raise ValueError(
                "preassay readout changed its causal root or request census"
            )


def seal_preassay_readout(
    source: ReactorBatchSource,
    references: RegimeReferenceComparators,
    causal: RegimeCausalPreparation,
    seal: RegimePredictionSeal,
    nomination: RegimeNominationPackage | None = None,
    calibration: RegimeCalibrationPackage | None = None,
) -> RegimePreassayReadout:
    if causal.role == "prospective" or (causal.role == "qualification") != (
        calibration is not None
    ):
        raise ValueError("qualification choices require the prior calibration receipt")
    if (seal.root, seal.role, seal.seed) != (
        causal.root,
        causal.role,
        causal.seed,
    ) or seal.causal_preparation != ObjectIdentity.from_record(
        f"{causal.root}.causal-preparation", causal
    ):
        raise ValueError("preassay readout substituted the sealed causal preparation")
    route = None if nomination is None else nomination.selected_route
    words = None
    valid = False
    if calibration is not None and nomination is not None:
        if calibration.nomination != ObjectIdentity.from_record(
            nomination.package_id, nomination
        ):
            raise ValueError("qualification screen substituted its nomination")
        if (
            route is not None
            and calibration.q_temperature is not None
            and calibration.q_cooling is not None
        ):
            selected = next(
                value for value in nomination.routes if value.route == route
            )
            if (
                selected.coefficient_model_id is None
                or selected.absolute_model_id is None
            ):
                raise ValueError("selected qualification route lacks frozen models")
            words = _model_forecast(
                seal,
                CONTEXTS.index(route),
                selected.coefficient_model_id,
                selected.absolute_model_id,
                float(calibration.q_temperature),
                float(calibration.q_cooling),
            )
            valid, _ = causal_preparation_validity(causal, route)
    decisions = (
        ()
        if words is None
        else tuple(
            select_word(
                request,
                words,
                law_valid=bool(calibration and calibration.precision_pass),
                preparation_safe=valid,
                clock_valid=valid,
                authority_valid=True,
                numerical_valid=True,
            )
            for request in REQUESTS_K
        )
    )
    return RegimePreassayReadout(
        causal.root,
        causal.role,
        ObjectIdentity.from_record(f"{causal.root}.prediction-seal", seal),
        ObjectIdentity.from_record("regime.reference-comparators", references),
        comparator_forecasts(source, references, causal),
        None
        if calibration is None
        else ObjectIdentity.from_record(calibration.package_id, calibration),
        route,
        valid,
        ()
        if words is None
        else tuple(ProspectiveWordForecast.from_word(v) for v in words),
        tuple(v.choice for v in decisions)
        if decisions
        else (None,) * (4 if calibration is not None else 0),
        tuple(v.reasons for v in decisions)
        if decisions
        else (("CALIBRATED_CAUSAL_CHART_UNAVAILABLE",),)
        * (4 if calibration is not None else 0),
    )


@dataclass(frozen=True, slots=True)
class RegimeComparatorError(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-comparator-error'
    root: str
    context: str
    comparator: str
    measured_valid: bool
    support: tuple[bool, bool, bool]
    nominal_peak_errors_K: tuple[D, ...]
    refined_peak_errors_K: tuple[D, ...]
    nominal_cooling_errors_K: tuple[D, ...]
    refined_cooling_errors_K: tuple[D, ...]


@dataclass(frozen=True, slots=True)
class RegimeOpportunityRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-opportunity-root'
    root: str
    decisions: ObjectIdentity
    preparation_safe: bool
    measured_valid: bool
    choices: tuple[int | None, ...]
    actual_feasible: tuple[bool, ...]
    chosen_successes: tuple[bool, ...]
    actual_cooling_headroom_K: tuple[D, ...]
    predicted_cooling_headroom_K: tuple[D, ...]
    temperature_upper_margins_K: tuple[D, ...]
    comparator_headroom_K: tuple[tuple[str, tuple[D, ...]], ...]


@dataclass(frozen=True, slots=True)
class RegimeOpportunityReadout(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/reactor-regime-response/regime-opportunity-readout'
    )
    role: str
    preassay_seals: tuple[ObjectIdentity, ...]
    comparator_errors: tuple[RegimeComparatorError, ...]
    qualification_opportunities: tuple[RegimeOpportunityRoot, ...]
    nomination_margins: tuple[RegimeNominationMargin, ...] = ()
    interpretation: str = "DESCRIPTIVE_QUALIFICATION_SCREEN_NOT_CONTROLLER_USE"

    def __post_init__(self) -> None:
        expected = tuple(root for root, role, _, _ in ROOTS if role == self.role)
        if (
            tuple(v.object_id for v in self.preassay_seals)
            != tuple(f"{root}.preassay-readout" for root in expected)
            or tuple((v.root, v.context, v.comparator) for v in self.comparator_errors)
            != tuple(
                (root, context, name)
                for root in expected
                for context in CONTEXTS
                for name in COMPARATORS
            )
            or tuple(v.root for v in self.qualification_opportunities)
            != (expected if self.role == "qualification" else ())
        ):
            raise ValueError(
                "opportunity readout lost assigned roots, windows or comparators"
            )
        if tuple((v.root, v.route) for v in self.nomination_margins) != (
            tuple(
                (root, route)
                for root in expected
                for route in ("prepared_t0", "c_q", "p_q")
            )
            if self.role == "nomination"
            else ()
        ):
            raise ValueError(
                "nomination opportunity margins lost their three-route census"
            )


@dataclass(frozen=True, slots=True)
class RegimeNominationMargin(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-nomination-margin'
    root: str
    route: str
    nomination: ObjectIdentity
    words: tuple[ProspectiveWordForecast, ...]
    provisional_choices: tuple[int | None, ...]
    cooling_lower_minus_request_K: tuple[tuple[D, ...], ...]
    temperature_limit_minus_upper_K: tuple[D, ...]
    comparator_headroom_K: tuple[tuple[str, tuple[D, ...]], ...]
    reasons: tuple[str, ...]


def evaluate_opportunities(
    role: str,
    rows: tuple[
        tuple[
            RegimeCausalPreparation,
            RegimePrivatePreparation,
            RegimePredictionSeal,
            RegimeAssayPanel,
            RegimePreassayReadout,
        ],
        ...,
    ],
    nomination: RegimeNominationPackage | None = None,
) -> RegimeOpportunityReadout:
    errors = []
    opportunities = []
    seals = []
    nomination_margins = []
    if role == "nomination" and nomination is None:
        raise ValueError("nomination margin screen lacks its frozen route operands")
    for causal, private, seal, assay, decisions in rows:
        if decisions.prediction_seal != ObjectIdentity.from_record(
            f"{causal.root}.prediction-seal", seal
        ):
            raise ValueError("opportunity screen changed its pre-label decision seal")
        sealed_id = ObjectIdentity.from_record(
            f"{causal.root}.preassay-readout", decisions
        )
        seals.append(sealed_id)
        charts = {row.context: row for row in measured_contexts(causal, assay)}
        for forecast in decisions.comparators:
            comparison_chart = charts[forecast.context]
            values = []
            for predicted, measured in (
                (forecast.peak_K, comparison_chart.nominal_peak_K),
                (forecast.peak_K, comparison_chart.refined_peak_K),
                (forecast.cooling_K, comparison_chart.nominal_cooling_K),
                (forecast.cooling_K, comparison_chart.refined_cooling_K),
            ):
                values.append(
                    ()
                    if not predicted or measured is None
                    else _numbers(np.asarray(predicted, dtype=float) - measured)
                )
            errors.append(
                RegimeComparatorError(
                    causal.root,
                    comparison_chart.context,
                    forecast.comparator,
                    comparison_chart.valid,
                    forecast.supported,
                    *values,
                )
            )
        if role == "nomination":
            assert nomination is not None
            for route_choice in nomination.routes:
                provisional = None
                if (
                    route_choice.coefficient_model_id is not None
                    and route_choice.absolute_model_id is not None
                    and route_choice.provisional_q_temperature is not None
                    and route_choice.provisional_q_cooling is not None
                ):
                    provisional = _model_forecast(
                        seal,
                        CONTEXTS.index(route_choice.route),
                        route_choice.coefficient_model_id,
                        route_choice.absolute_model_id,
                        float(route_choice.provisional_q_temperature),
                        float(route_choice.provisional_q_cooling),
                    )
                opportunity = next(
                    (
                        row
                        for row in route_choice.opportunities
                        if row.root == causal.root
                    ),
                    None,
                )
                headrooms = []
                for forecast in decisions.comparators:
                    if forecast.context != route_choice.route:
                        continue
                    admissible = [
                        word
                        for word in (1, 2)
                        if forecast.peak_K
                        and forecast.supported[word]
                        and forecast.peak_K[word] <= D("356.2")
                    ]
                    headrooms.append(
                        (
                            forecast.comparator,
                            ()
                            if not admissible
                            else _numbers(
                                [
                                    max(
                                        float(forecast.cooling_K[word])
                                        for word in admissible
                                    )
                                    - request
                                    for request in REQUESTS_K
                                ]
                            ),
                        )
                    )
                nomination_margins.append(
                    RegimeNominationMargin(
                        causal.root,
                        route_choice.route,
                        ObjectIdentity.from_record(nomination.package_id, nomination),
                        ()
                        if provisional is None
                        else tuple(
                            ProspectiveWordForecast.from_word(word)
                            for word in provisional
                        ),
                        (None,) * 4 if opportunity is None else opportunity.choices,
                        ()
                        if provisional is None
                        else tuple(
                            _numbers(
                                [
                                    word.cooling_lower_K - request
                                    for word in provisional[1:]
                                ]
                            )
                            for request in REQUESTS_K
                        ),
                        ()
                        if provisional is None
                        else _numbers(
                            [
                                TEMPERATURE_LIMIT_K - word.temperature_upper_K
                                for word in provisional
                            ]
                        ),
                        tuple(headrooms),
                        route_choice.reasons
                        if opportunity is None
                        else opportunity.reasons,
                    )
                )
        if role != "qualification":
            continue
        route = decisions.selected_route
        chart = None if route is None else charts[route]
        safe = route is not None and preparation_evidence(causal, private, route).safe
        feasible: tuple[bool, ...] = (False,) * 4
        successes: tuple[bool, ...] = (False,) * 4
        headroom: tuple[D, ...] = ()
        if (
            chart is not None
            and chart.nominal_cooling_K is not None
            and chart.refined_cooling_K is not None
            and chart.nominal_peak_K is not None
            and chart.refined_peak_K is not None
        ):
            cooling = np.minimum(chart.nominal_cooling_K, chart.refined_cooling_K)
            peaks = np.maximum(chart.nominal_peak_K, chart.refined_peak_K)
            allowed = tuple(
                word for word in (1, 2) if peaks[word] <= TEMPERATURE_LIMIT_K
            )
            feasible = tuple(
                bool(
                    safe and chart.valid and any(cooling[word] >= g for word in allowed)
                )
                for g in REQUESTS_K
            )
            successes = tuple(
                bool(safe and chart.valid and word in allowed and cooling[word] >= g)
                for g, word in zip(REQUESTS_K, decisions.choices, strict=True)
            )
            if allowed:
                headroom = _numbers(
                    [max(cooling[word] for word in allowed) - g for g in REQUESTS_K]
                )
        words = tuple(value.to_word() for value in decisions.primary_words)
        available = tuple(
            value
            for value in words[1:]
            if value.supported
            and value.projection_valid
            and value.temperature_upper_K <= TEMPERATURE_LIMIT_K
        )
        predicted_headroom = (
            ()
            if not available
            else _numbers(
                [max(v.cooling_lower_K for v in available) - g for g in REQUESTS_K]
            )
        )
        comparator_headroom = []
        for forecast in decisions.comparators:
            if forecast.context != route:
                continue
            indices = tuple(
                word
                for word in (1, 2)
                if forecast.peak_K
                and forecast.supported[word]
                and forecast.peak_K[word] <= D("356.2")
            )
            comparator_headroom.append(
                (
                    forecast.comparator,
                    ()
                    if not indices
                    else _numbers(
                        [
                            max(float(forecast.cooling_K[word]) for word in indices) - g
                            for g in REQUESTS_K
                        ]
                    ),
                )
            )
        opportunities.append(
            RegimeOpportunityRoot(
                causal.root,
                sealed_id,
                safe,
                bool(chart and chart.valid),
                decisions.choices,
                feasible,
                successes,
                headroom,
                predicted_headroom,
                _numbers([TEMPERATURE_LIMIT_K - v.temperature_upper_K for v in words]),
                tuple(comparator_headroom),
            )
        )
    return RegimeOpportunityReadout(
        role,
        tuple(seals),
        tuple(errors),
        tuple(opportunities),
        tuple(nomination_margins),
    )
