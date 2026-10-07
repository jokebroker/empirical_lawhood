"""Response summaries, typed thresholds, support, and prediction."""

from __future__ import annotations

from hashlib import sha256
from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .charts import CATEGORICAL_COORDINATES, CHART_COORDINATES, CausalChart
from ..quantum_scientific_order import ScientificOrder, scientific_order_ranks
from .contracts import Action, PreparationUnit, stable_json_bytes, stratum_allocation
from .receivers import ActionReceiver, future_receiver
from .source import FutureDraw


RESPONSE_COORDINATES = (
    "burst_a",
    "burst_b",
    "n_a",
    "n_a_squared",
    "z_a_ref",
    "n_total",
    "q_a",
    "terminal_energy",
    "work_abs",
    "transport_signed",
    "transport_absolute",
    "measurement_innovation",
)
PRIMARY_COORDINATES = ("burst_a", "n_a", "n_a_squared", "z_a_ref")
NUMERIC_FLOOR = 1e-12


@dataclass(frozen=True, slots=True)
class PrefixResponse:
    unit_id: str
    action: Action
    draws: int
    burst_a: float
    burst_b: float
    n_a: float
    n_a_squared: float
    z_a_ref: float
    n_total: float
    q_a: float
    terminal_energy: float
    work_abs: float
    transport_signed: float
    transport_absolute: float
    measurement_innovation: float
    maximum_continuity_residual: float
    valid: bool
    standard_errors: Mapping[str, float]

    def vector(self) -> NDArray[np.float64]:
        return np.asarray(
            [getattr(self, coordinate) for coordinate in RESPONSE_COORDINATES],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class ResponseDelta:
    unit_id: str
    action: Action
    values: Mapping[str, float]
    valid: bool

    def vector(self) -> NDArray[np.float64]:
        return np.asarray(
            [self.values[coordinate] for coordinate in RESPONSE_COORDINATES],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class QuantumTrajectoryResponseEffectMateriality:
    values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class QuantumTrajectoryLawPredictionTolerance:
    values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class QuantumTrajectoryLawFiberEquivalence:
    values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class QuantumTrajectoryAdmissionMargins:
    values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class QuantumTrajectoryProspectiveEfficacyMargins:
    values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class NumericalValidityFloors:
    values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ThresholdFamilies:
    response_effect: QuantumTrajectoryResponseEffectMateriality
    law_prediction: QuantumTrajectoryLawPredictionTolerance
    fiber: QuantumTrajectoryLawFiberEquivalence
    law_admission: QuantumTrajectoryAdmissionMargins
    prospective_efficacy: QuantumTrajectoryProspectiveEfficacyMargins
    numerical: NumericalValidityFloors

    def __post_init__(self) -> None:
        expected = set(RESPONSE_COORDINATES)
        for name, family in (
            ("response_effect", self.response_effect),
            ("chart_law", self.law_prediction),
            ("fiber", self.fiber),
        ):
            missing = expected - set(family.values)
            if missing:
                raise ValueError(f"{name} threshold family omits {sorted(missing)}")


@dataclass(frozen=True, slots=True)
class ChartScaler:
    chart_id: str
    coordinate_ids: tuple[str, ...]
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
    inference_latency_seconds: float


@dataclass(frozen=True, slots=True)
class DevelopmentHandoff:
    chart_id: str
    neighbor_count: int
    scaler: ChartScaler
    thresholds: ThresholdFamilies
    scores: tuple[CandidateScore, ...]
    donor_unit_ids: tuple[str, ...]
    reference_mean_a: float


def summarize_draws(
    draws: Sequence[FutureDraw],
    *,
    gamma: float,
    l_sites: int,
    particles: int,
    reference_mean_a: float = 0.0,
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
    arrays = {
        "burst_a": np.asarray([value.burst_a for value in receivers]),
        "burst_b": np.asarray([value.burst_b for value in receivers]),
        "n_a": np.asarray([value.n_a for value in receivers]),
        "n_a_squared": np.asarray([value.n_a_squared for value in receivers]),
        "n_total": np.asarray([value.n_total for value in receivers]),
        "q_a": np.asarray([draw.terminal_q_a for draw in draws]),
        "terminal_energy": np.asarray([draw.terminal_energy for draw in draws]),
        "work_abs": np.asarray([draw.work_abs for draw in draws]),
        "transport_signed": np.asarray([draw.transport.coherent_signed for draw in draws]),
        "transport_absolute": np.asarray([draw.transport.coherent_absolute for draw in draws]),
        "measurement_innovation": np.asarray(
            [draw.transport.measurement_innovation for draw in draws]
        ),
    }
    arrays["z_a_ref"] = (
        arrays["n_a_squared"] - (2.0 * reference_mean_a + 1.0) * arrays["n_a"] + reference_mean_a**2
    )
    standard_errors = {
        key: float(np.std(value, ddof=1) / math.sqrt(len(value))) for key, value in arrays.items()
    }
    return PrefixResponse(
        unit_id=next(iter(unit_ids)),
        action=next(iter(actions)),
        draws=len(draws),
        maximum_continuity_residual=max(abs(draw.transport.continuity_residual) for draw in draws),
        valid=all(
            draw.transport.valid
            and draw.maximum_norm_error <= 1e-11
            and draw.maximum_particle_number_error <= 1e-12
            and draw.ledger.delivery_discrepancy == 0
            for draw in draws
        ),
        standard_errors=standard_errors,
        **{key: float(value.mean()) for key, value in arrays.items()},
    )


def apply_reference_mean(
    response: PrefixResponse,
    reference_mean_a: float,
) -> PrefixResponse:
    z_value = (
        response.n_a_squared - (2.0 * reference_mean_a + 1.0) * response.n_a + reference_mean_a**2
    )
    return replace(response, z_a_ref=z_value)


def response_delta(action: PrefixResponse, hold: PrefixResponse) -> ResponseDelta:
    if (
        action.unit_id != hold.unit_id
        or action.action is Action.HOLD
        or hold.action is not Action.HOLD
    ):
        raise ValueError("response delta requires matched nonhold and hold")
    return ResponseDelta(
        unit_id=action.unit_id,
        action=action.action,
        values={
            coordinate: getattr(action, coordinate) - getattr(hold, coordinate)
            for coordinate in RESPONSE_COORDINATES
        },
        valid=action.valid and hold.valid,
    )


def split_development_units(
    units: Sequence[PreparationUnit],
    *,
    fit_size: int,
    scientific_unit_order: ScientificOrder | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ranks = scientific_order_ranks(scientific_unit_order, identifiers=tuple(unit.unit_id for unit in units))
    if not 0 < fit_size < len(units):
        raise ValueError("development split size is invalid")
    target = stratum_allocation(
        fit_size,
        l_sites=12,
        particles=6,
        minimum_per_stratum=8,
    )
    by_k: dict[int, list[PreparationUnit]] = {k: [] for k in range(7)}
    for unit in units:
        by_k[unit.k_left].append(unit)
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
        if len(local) <= target[k_left]:
            raise ValueError("development stratum cannot supply both split halves")
        fit.extend(unit.unit_id for unit in local[: target[k_left]])
        selection.extend(unit.unit_id for unit in local[target[k_left] :])
    if len(fit) != fit_size or set(fit) & set(selection):
        raise AssertionError("development split is invalid")
    return tuple(sorted(fit, key=ranks.__getitem__)), tuple(sorted(selection, key=ranks.__getitem__))


def derive_thresholds(
    *,
    fit_responses: Mapping[tuple[str, Action], PrefixResponse],
    hold_action_receiver: ActionReceiver,
) -> tuple[ThresholdFamilies, float]:
    holds = [
        value
        for (unit_id, action), value in fit_responses.items()
        if action is Action.HOLD and unit_id == value.unit_id
    ]
    nonzero = [value for (_, action), value in fit_responses.items() if action is not Action.HOLD]
    if not holds or len(nonzero) != 2 * len(holds):
        raise ValueError("threshold derivation requires complete three-action fit rows")

    def median(field: str, values: Sequence[PrefixResponse]) -> float:
        return float(np.median([getattr(value, field) for value in values]))

    reference_mean_a = median("n_a", holds)
    response_effect = {
        "burst_a": max(0.75, 0.15 * median("burst_a", holds)),
        "burst_b": max(0.75, 0.15 * median("burst_b", holds)),
        "n_a": max(1.0, 0.10 * median("n_a", holds)),
        "n_a_squared": max(3.0, 0.25 * abs(hold_action_receiver.operator.v_anom_a)),
        "z_a_ref": max(3.0, 0.25 * abs(hold_action_receiver.operator.v_anom_a)),
        "n_total": max(1.0, 0.10 * median("n_total", holds)),
        "q_a": 0.10,
        "terminal_energy": 0.50,
        "work_abs": max(0.10, 0.20 * median("work_abs", nonzero)),
        "transport_signed": max(0.10, 0.20 * median("transport_absolute", holds)),
        "transport_absolute": max(0.10, 0.20 * median("transport_absolute", holds)),
        "measurement_innovation": max(
            0.10, 0.20 * float(np.median(np.abs([value.measurement_innovation for value in holds])))
        ),
    }
    law_prediction = {
        "burst_a": max(0.50, 0.10 * median("burst_a", holds)),
        "burst_b": max(0.50, 0.10 * median("burst_b", holds)),
        "n_a": max(0.75, 0.075 * median("n_a", holds)),
        "n_a_squared": max(2.0, 0.15 * abs(median("n_a_squared", holds))),
        "z_a_ref": max(2.0, 0.15 * abs(median("z_a_ref", holds))),
        "n_total": max(0.75, 0.075 * median("n_total", holds)),
        "q_a": 0.075,
        "terminal_energy": 0.25,
        "work_abs": max(0.075, 0.15 * median("work_abs", nonzero)),
        "transport_signed": max(0.075, 0.15 * median("transport_absolute", holds)),
        "transport_absolute": max(0.075, 0.15 * median("transport_absolute", holds)),
        "measurement_innovation": max(
            0.075,
            0.15 * float(np.median(np.abs([value.measurement_innovation for value in holds]))),
        ),
    }
    fiber = {coordinate: 0.5 * response_effect[coordinate] for coordinate in RESPONSE_COORDINATES}
    families = ThresholdFamilies(
        response_effect=QuantumTrajectoryResponseEffectMateriality(response_effect),
        law_prediction=QuantumTrajectoryLawPredictionTolerance(law_prediction),
        fiber=QuantumTrajectoryLawFiberEquivalence(fiber),
        law_admission=QuantumTrajectoryAdmissionMargins(
            {
                "transport_activity_ratio": 0.75,
                "continuity_tolerance": 1e-8,
                "mean_q_a": response_effect["q_a"],
                "burst_b": response_effect["burst_b"],
                "terminal_energy": response_effect["terminal_energy"],
                "work_abs": response_effect["work_abs"],
            }
        ),
        prospective_efficacy=QuantumTrajectoryProspectiveEfficacyMargins(
            {
                "burst_a": response_effect["burst_a"],
                "v_anom_a": response_effect["z_a_ref"],
                "fano_a": 0.10,
            }
        ),
        numerical=NumericalValidityFloors(
            {
                "norm": 1e-11,
                "particle_number": 1e-12,
                "distance": 1e-12,
                "continuity": 1e-8,
            }
        ),
    )
    return families, reference_mean_a


def fit_chart_scaler(
    *,
    chart_id: str,
    charts: Mapping[str, CausalChart],
    neighbor_counts: Sequence[int],
    support_quantile: float,
    scientific_unit_order: ScientificOrder | None = None,
) -> ChartScaler:
    ranks = scientific_order_ranks(scientific_unit_order, identifiers=tuple(charts))
    ordered = [charts[unit_id] for unit_id in sorted(charts, key=ranks.__getitem__)]
    if not ordered or any(chart.chart_id != chart_id for chart in ordered):
        raise ValueError("chart scaler input differs")
    coordinate_ids = ordered[0].coordinate_ids
    if any(chart.coordinate_ids != coordinate_ids for chart in ordered):
        raise ValueError("chart coordinate order differs")
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for index, coordinate in enumerate(coordinate_ids):
        if coordinate in CATEGORICAL_COORDINATES:
            continue
        values = np.asarray([float(chart.values[index]) for chart in ordered])
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        floor = 0.1 if "sin" in coordinate or "cos" in coordinate else 1e-3
        centers[coordinate] = center
        scales[coordinate] = max(mad, floor)
    provisional = ChartScaler(chart_id, coordinate_ids, centers, scales, {})
    matrix = np.asarray(
        [
            [
                chart_distance(left, right, provisional) if left_index != right_index else np.inf
                for right_index, right in enumerate(ordered)
            ]
            for left_index, left in enumerate(ordered)
        ]
    )
    support: dict[int, float] = {}
    for neighbor_count in neighbor_counts:
        if neighbor_count >= len(ordered):
            raise ValueError("neighbour count exceeds the fit roster")
        kth = np.partition(matrix, neighbor_count - 1, axis=1)[:, neighbor_count - 1]
        support[neighbor_count] = float(np.quantile(kth, support_quantile, method="higher"))
    return replace(provisional, support_radius_by_k=support, scientific_order_sha256=sha256(stable_json_bytes(scientific_unit_order)).hexdigest())


def chart_distance(
    left: CausalChart,
    right: CausalChart,
    scaler: ChartScaler,
) -> float:
    if (
        left.chart_id != scaler.chart_id
        or right.chart_id != scaler.chart_id
        or left.coordinate_ids != scaler.coordinate_ids
        or right.coordinate_ids != scaler.coordinate_ids
    ):
        raise ValueError("chart distance operands differ")
    distances: list[float] = []
    for coordinate, left_value, right_value in zip(
        scaler.coordinate_ids,
        left.values,
        right.values,
        strict=True,
    ):
        if coordinate in CATEGORICAL_COORDINATES:
            distances.append(0.0 if left_value == right_value else 1.0)
        else:
            distances.append(
                abs(float(left_value) - float(right_value)) / scaler.scales[coordinate]
            )
    return max(distances, default=0.0)


def predict_response(
    *,
    target_chart: CausalChart,
    target_delta: ResponseDelta,
    donor_charts: Mapping[str, CausalChart],
    donor_deltas: Mapping[tuple[str, Action], ResponseDelta],
    scaler: ChartScaler,
    neighbor_count: int,
    tolerances: QuantumTrajectoryLawPredictionTolerance,
    scientific_donor_order: ScientificOrder | None = None,
) -> PredictionRow:
    donor_ranks = scientific_order_ranks(scientific_donor_order, identifiers=tuple(donor_charts))
    if any(unit_id not in donor_ranks for unit_id, _action in donor_deltas):
        raise ValueError("donor response catalogue differs from the declared scientific order")
    scientific_digest = sha256(stable_json_bytes(scientific_donor_order)).hexdigest()
    candidates = []
    for (unit_id, action), delta in donor_deltas.items():
        if action is not target_delta.action or unit_id == target_delta.unit_id:
            continue
        distance = chart_distance(target_chart, donor_charts[unit_id], scaler)
        candidates.append((distance, unit_id, delta))
    candidates.sort(key=lambda item: (item[0], donor_ranks[item[1]]))
    if len(candidates) < neighbor_count:
        raise ValueError("predictor lacks enough action-matched donors")
    neighbors = candidates[:neighbor_count]
    kth_distance = neighbors[-1][0]
    inside = kth_distance <= scaler.support_radius_by_k[neighbor_count]
    observed = target_delta.vector()
    if not inside:
        return PredictionRow(
            target_unit_id=target_delta.unit_id,
            action=target_delta.action,
            chart_id=target_chart.chart_id,
            neighbor_count=neighbor_count,
            inside_support=False,
            kth_distance=kth_distance,
            predicted=None,
            observed=observed,
            errors=None,
            adequate=None,
            primary_adequate=None,
            donor_unit_ids=tuple(item[1] for item in neighbors),
            scientific_order_sha256=scientific_digest,
        )
    predicted = np.mean([item[2].vector() for item in neighbors], axis=0)
    errors = np.abs(predicted - observed)
    limits = np.asarray([tolerances.values[coordinate] for coordinate in RESPONSE_COORDINATES])
    primary_indices = [RESPONSE_COORDINATES.index(value) for value in PRIMARY_COORDINATES]
    return PredictionRow(
        target_unit_id=target_delta.unit_id,
        action=target_delta.action,
        chart_id=target_chart.chart_id,
        neighbor_count=neighbor_count,
        inside_support=True,
        kth_distance=kth_distance,
        predicted=np.asarray(predicted),
        observed=observed,
        errors=errors,
        adequate=bool(np.all(errors <= limits)) and target_delta.valid,
        primary_adequate=bool(np.all(errors[primary_indices] <= limits[primary_indices]))
        and target_delta.valid,
        donor_unit_ids=tuple(item[1] for item in neighbors),
        scientific_order_sha256=scientific_digest,
    )


def score_candidate(
    rows: Sequence[PredictionRow],
    *,
    tolerances: QuantumTrajectoryLawPredictionTolerance,
    inference_latency_seconds: float,
) -> CandidateScore:
    if not rows:
        raise ValueError("candidate score requires prediction rows")
    supported = [row for row in rows if row.inside_support]
    primary_losses = sum(row.primary_adequate is False for row in supported)
    full_losses = sum(row.adequate is False for row in supported)
    scaled: list[float] = []
    limits = np.asarray([tolerances.values[coordinate] for coordinate in RESPONSE_COORDINATES])
    for row in supported:
        if row.errors is not None:
            scaled.append(float(np.max(row.errors / np.maximum(limits, NUMERIC_FLOOR))))
    exemplar = rows[0]
    return CandidateScore(
        chart_id=exemplar.chart_id,
        neighbor_count=exemplar.neighbor_count,
        support_fraction=len(supported) / len(rows),
        primary_loss_fraction=primary_losses / len(supported) if supported else 1.0,
        full_loss_fraction=full_losses / len(supported) if supported else 1.0,
        q95_scaled_error=float(np.quantile(scaled, 0.95)) if scaled else math.inf,
        maximum_scaled_error=max(scaled, default=math.inf),
        chart_dimension=len(CHART_COORDINATES[exemplar.chart_id]),
        inference_latency_seconds=inference_latency_seconds,
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
        raise ValueError("candidate selection requires a score table")
    maximum_support = max(score.support_fraction for score in scores)
    retained = [score for score in scores if score.support_fraction >= maximum_support - 0.05]
    minimum_loss = min(score.full_loss_fraction for score in retained)
    retained = [score for score in retained if score.full_loss_fraction <= minimum_loss + 0.05]
    return min(
        retained,
        key=lambda score: (
            score.chart_dimension,
            score.neighbor_count,
            score.q95_scaled_error,
            score.inference_latency_seconds,
            chart_ranks[score.chart_id],
        ),
    )


__all__ = [
    "CandidateScore",
    "ChartScaler",
    "DevelopmentHandoff",
    "NumericalValidityFloors",
    'QuantumTrajectoryResponseEffectMateriality',
    'QuantumTrajectoryLawFiberEquivalence',
    'QuantumTrajectoryLawPredictionTolerance',
    'QuantumTrajectoryAdmissionMargins',
    'QuantumTrajectoryProspectiveEfficacyMargins',
    "PRIMARY_COORDINATES",
    "PrefixResponse",
    "PredictionRow",
    "RESPONSE_COORDINATES",
    "ResponseDelta",
    "ThresholdFamilies",
    "apply_reference_mean",
    "chart_distance",
    "derive_thresholds",
    "fit_chart_scaler",
    "predict_response",
    "response_delta",
    "score_candidate",
    "select_candidate",
    "split_development_units",
    "summarize_draws",
]
