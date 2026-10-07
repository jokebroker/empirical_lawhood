"""Development response catalogue, chart selection and prediction."""

from __future__ import annotations

from hashlib import sha256
from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..quantum_scientific_order import ScientificOrder, scientific_order_ranks
from .contracts import Action, PreparationUnit, stable_json_bytes, stratum_allocation
from .receivers import CHART_COORDINATES, ActionReceiver, future_receiver
from .source import FutureDraw


REQUIRED_RESPONSE_IDS = ("burst_a", "burst_b", "n_a", "q_a", "work_abs")
NUMERIC_FLOOR = 1e-12


@dataclass(frozen=True, slots=True)
class PrefixResponse:
    unit_id: str
    action: Action
    draws: int
    burst_a: float
    burst_b: float
    n_a: float
    var_n_a: float
    n_total: float
    q_a: float
    work_abs: float
    standard_errors: Mapping[str, float]

    def required_vector(self) -> NDArray[np.float64]:
        return np.asarray(
            [self.burst_a, self.burst_b, self.n_a, self.q_a, self.work_abs],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class ResponseDelta:
    unit_id: str
    action: Action
    burst_a: float
    burst_b: float
    n_a: float
    q_a: float
    work_abs: float
    var_n_a: float

    def required_vector(self) -> NDArray[np.float64]:
        return np.asarray(
            [self.burst_a, self.burst_b, self.n_a, self.q_a, self.work_abs],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class EffectThresholds:
    values: Mapping[str, float]

    def __getitem__(self, key: str) -> float:
        return self.values[key]


@dataclass(frozen=True, slots=True)
class PredictionTolerances:
    values: Mapping[str, float]

    def __getitem__(self, key: str) -> float:
        return self.values[key]


@dataclass(frozen=True, slots=True)
class FiberMargins:
    values: Mapping[str, float]

    def __getitem__(self, key: str) -> float:
        return self.values[key]


@dataclass(frozen=True, slots=True)
class ThresholdFamilies:
    delta: EffectThresholds
    prediction: PredictionTolerances
    fiber: FiberMargins

    def __post_init__(self) -> None:
        if not isinstance(self.delta, EffectThresholds):
            raise TypeError("response effect thresholds require their typed family")
        if not isinstance(self.prediction, PredictionTolerances):
            raise TypeError("response prediction tolerances require their typed family")
        if not isinstance(self.fiber, FiberMargins):
            raise TypeError("natural-fiber margins require their typed family")


@dataclass(frozen=True, slots=True)
class ChartScaler:
    chart_id: str
    coordinate_ids: tuple[str, ...]
    numeric_coordinate_ids: tuple[str, ...]
    categorical_coordinate_ids: tuple[str, ...]
    centers: Mapping[str, float]
    scales: Mapping[str, float]
    support_radius_by_k: Mapping[int, float]
    scientific_order_sha256: str = ""


@dataclass(frozen=True, slots=True)
class PredictionRow:
    target_unit_id: str
    action: Action
    chart_id: str
    neighbor_count: int
    inside_support: bool
    kth_distance: float
    predicted: NDArray[np.float64] | None
    observed: NDArray[np.float64]
    errors: NDArray[np.float64] | None
    adequate: bool | None
    primary_adequate: bool | None
    donor_unit_ids: tuple[str, ...]
    scientific_order_sha256: str = ""


@dataclass(frozen=True, slots=True)
class CandidateScore:
    chart_id: str
    neighbor_count: int
    support_fraction: float
    primary_loss_fraction: float
    full_loss_fraction: float
    q95_scaled_error: float
    maximum_scaled_error: float
    chart_dimension: int


@dataclass(frozen=True, slots=True)
class DevelopmentHandoff:
    chart_id: str
    neighbor_count: int
    scaler: ChartScaler
    thresholds: ThresholdFamilies
    scores: tuple[CandidateScore, ...]
    donor_unit_ids: tuple[str, ...]
    scientific_order_sha256: str = ""


def summarize_draws(
    draws: Sequence[FutureDraw],
    *,
    gamma: float,
    l_sites: int,
    particles: int,
) -> PrefixResponse:
    if len(draws) < 2:
        raise ValueError("prefix response requires repeated future draws")
    unit_ids = {draw.unit_id for draw in draws}
    actions = {draw.action for draw in draws}
    if len(unit_ids) != 1 or len(actions) != 1:
        raise ValueError("prefix response draws cross unit or action")
    receivers = [
        future_receiver(
            draw,
            gamma=gamma,
            l_sites=l_sites,
            particles=particles,
        )
        for draw in draws
    ]
    rows = {
        "burst_a": np.asarray([value.burst_a for value in receivers]),
        "burst_b": np.asarray([value.burst_b for value in receivers]),
        "n_a": np.asarray([value.n_a for value in receivers]),
        "n_total": np.asarray([value.n_total for value in receivers]),
        "q_a": np.asarray([draw.terminal_q_a for draw in draws]),
        "work_abs": np.asarray([draw.work_abs for draw in draws]),
    }
    standard_errors = {
        key: float(np.std(value, ddof=1) / math.sqrt(len(value))) for key, value in rows.items()
    }
    return PrefixResponse(
        unit_id=next(iter(unit_ids)),
        action=next(iter(actions)),
        draws=len(draws),
        burst_a=float(rows["burst_a"].mean()),
        burst_b=float(rows["burst_b"].mean()),
        n_a=float(rows["n_a"].mean()),
        var_n_a=float(np.var(rows["n_a"], ddof=1)),
        n_total=float(rows["n_total"].mean()),
        q_a=float(rows["q_a"].mean()),
        work_abs=float(rows["work_abs"].mean()),
        standard_errors=standard_errors,
    )


def response_delta(action: PrefixResponse, hold: PrefixResponse) -> ResponseDelta:
    if (
        action.unit_id != hold.unit_id
        or action.action is Action.HOLD
        or hold.action is not Action.HOLD
    ):
        raise ValueError("response delta requires matched action and hold")
    return ResponseDelta(
        unit_id=action.unit_id,
        action=action.action,
        burst_a=action.burst_a - hold.burst_a,
        burst_b=action.burst_b - hold.burst_b,
        n_a=action.n_a - hold.n_a,
        q_a=action.q_a - hold.q_a,
        work_abs=action.work_abs - hold.work_abs,
        var_n_a=action.var_n_a - hold.var_n_a,
    )


def split_development_units(
    units: Sequence[PreparationUnit],
    *,
    scientific_unit_order: ScientificOrder | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ranks = scientific_order_ranks(scientific_unit_order, identifiers=tuple(unit.unit_id for unit in units))
    if len(units) != 96:
        raise ValueError("development split requires exactly 96 units")
    by_k: dict[int, list[PreparationUnit]] = {}
    for unit in units:
        by_k.setdefault(unit.k_left, []).append(unit)
    target = stratum_allocation(
        total_units=48,
        l_sites=12,
        particles=6,
        minimum_per_stratum=2,
    )
    fit: list[str] = []
    selection: list[str] = []
    for k_left in range(7):
        local = sorted(
            by_k[k_left],
            key=lambda unit: (
                unit.boundary_occupation,
                unit.translation_orbit,
                ranks[unit.unit_id],
            ),
        )
        fit.extend(unit.unit_id for unit in local[: target[k_left]])
        selection.extend(unit.unit_id for unit in local[target[k_left] :])
    if len(fit) != 48 or len(selection) != 48 or set(fit) & set(selection):
        raise AssertionError("development fit/selection split is invalid")
    return tuple(sorted(fit, key=ranks.__getitem__)), tuple(sorted(selection, key=ranks.__getitem__))


def derive_thresholds(
    *,
    fit_responses: Mapping[tuple[str, Action], PrefixResponse],
    hold_action_receiver: ActionReceiver,
) -> ThresholdFamilies:
    hold = [
        value
        for (unit_id, action), value in fit_responses.items()
        if action is Action.HOLD and unit_id == value.unit_id
    ]
    nonzero = [value for (_, action), value in fit_responses.items() if action is not Action.HOLD]
    if len(hold) != 48 or len(nonzero) != 96:
        raise ValueError("threshold derivation requires the complete development fit set")

    def median(field: str, values: Sequence[PrefixResponse]) -> float:
        return float(np.median([getattr(value, field) for value in values]))

    delta = {
        "burst_a": max(0.75, 0.15 * median("burst_a", hold)),
        "burst_b": max(0.75, 0.15 * median("burst_b", hold)),
        "n_a": max(1.0, 0.10 * median("n_a", hold)),
        "v_anom_a": max(
            3.0,
            0.25 * abs(hold_action_receiver.operator.v_anom_a),
        ),
        "fano_a": 0.10,
        "q_a": 0.10,
        "work_abs": max(0.10, 0.20 * median("work_abs", nonzero)),
    }
    prediction = {
        "burst_a": max(0.50, 0.10 * median("burst_a", hold)),
        "burst_b": max(0.50, 0.10 * median("burst_b", hold)),
        "n_a": max(0.75, 0.075 * median("n_a", hold)),
        "q_a": 0.075,
        "work_abs": max(0.075, 0.15 * median("work_abs", nonzero)),
    }
    fiber = {
        key: 0.5
        * delta[
            {
                "burst_a": "burst_a",
                "burst_b": "burst_b",
                "n_a": "n_a",
                "q_a": "q_a",
                "work_abs": "work_abs",
            }[key]
        ]
        for key in REQUIRED_RESPONSE_IDS
    }
    return ThresholdFamilies(
        delta=EffectThresholds(delta),
        prediction=PredictionTolerances(prediction),
        fiber=FiberMargins(fiber),
    )


def build_scaler(
    *,
    chart_id: str,
    fit_charts: Mapping[str, Mapping[str, float | str]],
    neighbor_counts: Sequence[int] = (8, 16),
    scientific_unit_order: ScientificOrder | None = None,
) -> ChartScaler:
    ranks = scientific_order_ranks(scientific_unit_order, identifiers=tuple(fit_charts))
    ids = sorted(fit_charts, key=ranks.__getitem__)
    scientific_digest = sha256(stable_json_bytes(scientific_unit_order)).hexdigest()
    coordinates = CHART_COORDINATES[chart_id]
    numeric = tuple(coordinate for coordinate in coordinates if coordinate != "last_event_region")
    categorical = tuple(
        coordinate for coordinate in coordinates if coordinate == "last_event_region"
    )
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for coordinate in numeric:
        values = np.asarray(
            [float(fit_charts[unit_id][coordinate]) for unit_id in ids],
            dtype=np.float64,
        )
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        centers[coordinate] = center
        scales[coordinate] = max(mad, NUMERIC_FLOOR)
    provisional = ChartScaler(
        chart_id=chart_id,
        coordinate_ids=coordinates,
        numeric_coordinate_ids=numeric,
        categorical_coordinate_ids=categorical,
        centers=centers,
        scales=scales,
        support_radius_by_k={},
        scientific_order_sha256=scientific_digest,
    )
    neighbor_distances: dict[int, list[float]] = {count: [] for count in neighbor_counts}
    for target_id in ids:
        distances = sorted(
            chart_distance(
                fit_charts[target_id],
                fit_charts[donor_id],
                provisional,
            )
            for donor_id in ids
            if donor_id != target_id
        )
        for count in neighbor_counts:
            if len(distances) < count:
                raise ValueError("fit catalogue is smaller than neighbor count")
            neighbor_distances[count].append(distances[count - 1])
    radii = {
        count: float(
            np.quantile(
                values,
                0.95,
                method="higher",
            )
        )
        for count, values in neighbor_distances.items()
    }
    return ChartScaler(
        chart_id=chart_id,
        coordinate_ids=coordinates,
        numeric_coordinate_ids=numeric,
        categorical_coordinate_ids=categorical,
        centers=centers,
        scales=scales,
        support_radius_by_k=radii,
        scientific_order_sha256=scientific_digest,
    )


def chart_distance(
    left: Mapping[str, float | str],
    right: Mapping[str, float | str],
    scaler: ChartScaler,
) -> float:
    distances = [
        abs(float(left[coordinate]) - float(right[coordinate])) / scaler.scales[coordinate]
        for coordinate in scaler.numeric_coordinate_ids
    ]
    distances.extend(
        0.0 if left[coordinate] == right[coordinate] else 1.0
        for coordinate in scaler.categorical_coordinate_ids
    )
    return max(distances, default=0.0)


def predict_rows(
    *,
    target_unit_ids: Sequence[str],
    donor_unit_ids: Sequence[str],
    charts: Mapping[str, Mapping[str, float | str]],
    deltas: Mapping[tuple[str, Action], ResponseDelta],
    scaler: ChartScaler,
    neighbor_count: int,
    thresholds: ThresholdFamilies,
    scientific_target_order: ScientificOrder | None = None,
    scientific_donor_order: ScientificOrder | None = None,
) -> tuple[PredictionRow, ...]:
    target_ranks = scientific_order_ranks(scientific_target_order, identifiers=tuple(target_unit_ids))
    donor_ranks = scientific_order_ranks(scientific_donor_order, identifiers=tuple(donor_unit_ids))
    scientific_digest = sha256(stable_json_bytes({
        "targets": scientific_target_order, "donors": scientific_donor_order,
    })).hexdigest()
    rows: list[PredictionRow] = []
    tolerance = np.asarray(
        [thresholds.prediction[key] for key in REQUIRED_RESPONSE_IDS],
        dtype=np.float64,
    )
    for target_id in sorted(target_unit_ids, key=target_ranks.__getitem__):
        for action in (Action.MINUS, Action.PLUS):
            candidates = sorted(
                (
                    (
                        chart_distance(charts[target_id], charts[donor_id], scaler),
                        donor_id,
                    )
                    for donor_id in donor_unit_ids
                    if donor_id != target_id and (donor_id, action) in deltas
                ),
                key=lambda value: (value[0], donor_ranks[value[1]]),
            )
            if len(candidates) < neighbor_count:
                raise ValueError("prediction catalogue lacks required neighbors")
            local = candidates[:neighbor_count]
            kth_distance = local[-1][0]
            observed = deltas[(target_id, action)].required_vector()
            if kth_distance > scaler.support_radius_by_k[neighbor_count]:
                rows.append(
                    PredictionRow(
                        target_unit_id=target_id,
                        action=action,
                        chart_id=scaler.chart_id,
                        neighbor_count=neighbor_count,
                        inside_support=False,
                        kth_distance=kth_distance,
                        predicted=None,
                        observed=observed,
                        errors=None,
                        adequate=None,
                        primary_adequate=None,
                        donor_unit_ids=tuple(donor for _, donor in local),
                        scientific_order_sha256=scientific_digest,
                    )
                )
                continue
            predicted = np.mean(
                [deltas[(donor, action)].required_vector() for _, donor in local],
                axis=0,
            )
            errors = np.abs(predicted - observed)
            rows.append(
                PredictionRow(
                    target_unit_id=target_id,
                    action=action,
                    chart_id=scaler.chart_id,
                    neighbor_count=neighbor_count,
                    inside_support=True,
                    kth_distance=kth_distance,
                    predicted=predicted,
                    observed=observed,
                    errors=errors,
                    adequate=bool(np.all(errors <= tolerance)),
                    primary_adequate=bool(errors[0] <= tolerance[0]),
                    donor_unit_ids=tuple(donor for _, donor in local),
                    scientific_order_sha256=scientific_digest,
                )
            )
    return tuple(rows)


def score_candidate(
    rows: Sequence[PredictionRow],
    *,
    thresholds: ThresholdFamilies,
) -> CandidateScore:
    if not rows:
        raise ValueError("candidate score lacks prediction rows")
    supported = [row for row in rows if row.inside_support]
    tolerance = np.asarray(
        [thresholds.prediction[key] for key in REQUIRED_RESPONSE_IDS],
        dtype=np.float64,
    )
    scaled = [float(np.max(row.errors / tolerance)) for row in supported if row.errors is not None]
    return CandidateScore(
        chart_id=rows[0].chart_id,
        neighbor_count=rows[0].neighbor_count,
        support_fraction=len(supported) / len(rows),
        primary_loss_fraction=(
            sum(row.primary_adequate is False for row in supported) / len(supported)
            if supported
            else 1.0
        ),
        full_loss_fraction=(
            sum(row.adequate is False for row in supported) / len(supported) if supported else 1.0
        ),
        q95_scaled_error=(
            float(np.quantile(scaled, 0.95, method="higher")) if scaled else math.inf
        ),
        maximum_scaled_error=max(scaled, default=math.inf),
        chart_dimension=len(CHART_COORDINATES[rows[0].chart_id]),
    )


def select_candidate(
    scores: Sequence[CandidateScore],
    *,
    scientific_chart_order: ScientificOrder | None = None,
) -> CandidateScore:
    chart_ids = tuple(dict.fromkeys(score.chart_id for score in scores))
    chart_ranks = scientific_order_ranks(scientific_chart_order, identifiers=chart_ids)
    if len({(score.chart_id, score.neighbor_count) for score in scores}) != len(scores):
        raise ValueError("candidate table repeats a chart/neighbor cell")
    if not scores:
        raise ValueError("chart selection lacks candidates")
    maximum_support = max(score.support_fraction for score in scores)
    retained = [score for score in scores if score.support_fraction >= maximum_support - 0.05]
    minimum_primary_loss = min(score.primary_loss_fraction for score in retained)
    retained = [
        score for score in retained if score.primary_loss_fraction <= minimum_primary_loss + 0.05
    ]
    return min(
        retained,
        key=lambda score: (
            score.chart_dimension,
            score.neighbor_count,
            score.q95_scaled_error,
            chart_ranks[score.chart_id],
        ),
    )


def compile_development_handoff(
    *,
    fit_unit_ids: Sequence[str],
    selection_unit_ids: Sequence[str],
    charts: Mapping[str, Mapping[str, float | str]],
    deltas: Mapping[tuple[str, Action], ResponseDelta],
    thresholds: ThresholdFamilies,
    scientific_unit_order: ScientificOrder | None = None,
    scientific_chart_order: ScientificOrder | None = None,
) -> DevelopmentHandoff:
    ranks = scientific_order_ranks(scientific_unit_order, identifiers=tuple((*fit_unit_ids, *selection_unit_ids)))
    chart_ranks = scientific_order_ranks(scientific_chart_order, identifiers=tuple(CHART_COORDINATES))
    fit_order: ScientificOrder = tuple((label, ranks[label]) for label in fit_unit_ids)
    selection_order: ScientificOrder = tuple((label, ranks[label]) for label in selection_unit_ids)
    scores: list[CandidateScore] = []
    scalers: dict[str, ChartScaler] = {}
    for chart_id in sorted(CHART_COORDINATES, key=chart_ranks.__getitem__):
        scaler = build_scaler(
            chart_id=chart_id,
            fit_charts={unit_id: charts[unit_id] for unit_id in fit_unit_ids},
            scientific_unit_order=fit_order,
        )
        scalers[chart_id] = scaler
        for neighbor_count in (8, 16):
            rows = predict_rows(
                target_unit_ids=selection_unit_ids,
                scientific_target_order=selection_order,
                scientific_donor_order=fit_order,
                donor_unit_ids=fit_unit_ids,
                charts=charts,
                deltas=deltas,
                scaler=scaler,
                neighbor_count=neighbor_count,
                thresholds=thresholds,
            )
            scores.append(score_candidate(rows, thresholds=thresholds))
    candidate_order: ScientificOrder = tuple(
        (chart_id, chart_ranks[chart_id]) for chart_id in dict.fromkeys(score.chart_id for score in scores)
    )
    selected = select_candidate(scores, scientific_chart_order=candidate_order)
    return DevelopmentHandoff(
        chart_id=selected.chart_id,
        neighbor_count=selected.neighbor_count,
        scaler=scalers[selected.chart_id],
        thresholds=thresholds,
        scores=tuple(sorted(scores, key=lambda row: (chart_ranks[row.chart_id], row.neighbor_count))),
        donor_unit_ids=tuple(sorted((*fit_unit_ids, *selection_unit_ids), key=ranks.__getitem__)),
        scientific_order_sha256=sha256(stable_json_bytes({"units": scientific_unit_order, "charts": scientific_chart_order})).hexdigest(),
    )


def prospective_response_evaluation_units(
    *,
    deltas: Mapping[tuple[str, Action], ResponseDelta],
    thresholds: ThresholdFamilies,
    base_units: int = 96,
    block_units: int = 48,
    max_units: int = 192,
    scientific_unit_order: ScientificOrder | None = None,
) -> tuple[int, Mapping[str, float]]:
    """Choose independent-unit count from variance magnitude only."""

    unit_ids = tuple(dict.fromkeys(unit_id for unit_id, _action in deltas))
    ranks = scientific_order_ranks(scientific_unit_order, identifiers=unit_ids)
    z_value = NormalDist().inv_cdf(1.0 - 0.05 / (2.0 * 10.0))
    requirements: dict[str, float] = {}
    for coordinate_index, coordinate in enumerate(REQUIRED_RESPONSE_IDS):
        for action in (Action.MINUS, Action.PLUS):
            values = np.asarray(
                [
                    deltas[(unit_id, action)].required_vector()[coordinate_index]
                    for unit_id in sorted(unit_ids, key=ranks.__getitem__)
                    if (unit_id, action) in deltas
                ],
                dtype=np.float64,
            )
            if len(values) < 2:
                raise ValueError("prediction precision design lacks independent prefixes")
            standard_deviation = float(np.std(values, ddof=1))
            target_halfwidth = 0.5 * min(
                thresholds.delta[coordinate],
                thresholds.prediction[coordinate],
            )
            requirements[f"{coordinate}:{action.value}"] = (
                (z_value * standard_deviation / target_halfwidth) ** 2
                if target_halfwidth > 0
                else math.inf
            )
    required = max(requirements.values(), default=base_units)
    selected = base_units
    while selected < required and selected < max_units:
        selected += block_units
    return min(selected, max_units), requirements


__all__ = [
    "CandidateScore",
    "ChartScaler",
    "DevelopmentHandoff",
    "EffectThresholds",
    "FiberMargins",
    "NUMERIC_FLOOR",
    "PredictionRow",
    "PrefixResponse",
    "PredictionTolerances",
    "REQUIRED_RESPONSE_IDS",
    "ResponseDelta",
    "ThresholdFamilies",
    "build_scaler",
    "chart_distance",
    "compile_development_handoff",
    "derive_thresholds",
    "predict_rows",
    "prospective_response_evaluation_units",
    "response_delta",
    "score_candidate",
    "select_candidate",
    "split_development_units",
    "summarize_draws",
]
