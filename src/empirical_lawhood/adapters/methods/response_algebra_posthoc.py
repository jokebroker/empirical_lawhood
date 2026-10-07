"""Outcome-visible, complete-unit response-algebra trajectory methods.

The methods in this module are intentionally substrate-neutral.  Thin adapters
construct :class:`TrajectoryPanel` objects from receipt-verified episodes; the
estimators operate only on those panels.  Rows, words, clocks and numerical
views are nested under the physical preparation and never treated as
replicates.

These are exploratory finite-word methods.  They do not estimate generators,
Lie brackets, context-free substrate laws or probabilistic Fisher information.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from itertools import combinations
from typing import ClassVar, Final, Mapping

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)


WORD_IDS: Final = (
    "a-early",
    "a-late",
    "a-repeat",
    "a-then-b",
    "b-early",
    "b-late",
    "b-repeat",
    "b-then-a",
    "identity",
    "simultaneous-a-b",
)
NONIDENTITY_WORD_IDS: Final = tuple(value for value in WORD_IDS if value != "identity")


class PosthocDisposition(StrEnum):
    """Closed exploratory dispositions; no scalar score substitutes for them."""

    RESOLVED = "RESOLVED"
    BELOW_NUMERICAL_RESOLUTION = "BELOW_NUMERICAL_RESOLUTION"
    EQUIVALENT_WITHIN_ENVELOPE = "EQUIVALENT_WITHIN_ENVELOPE"
    ANNULAR = "ANNULAR"
    MATERIAL = "MATERIAL"
    UNEVALUABLE_PORT_SPAN_UNRESOLVED = "UNEVALUABLE_PORT_SPAN_UNRESOLVED"
    UNEVALUABLE_NUMERICAL_PAIR_ABSENT = "UNEVALUABLE_NUMERICAL_PAIR_ABSENT"
    UNEVALUABLE_PROJECTION_LIMITED = "UNEVALUABLE_PROJECTION_LIMITED"
    UNEVALUABLE_OPERAND_ABSENT = "UNEVALUABLE_OPERAND_ABSENT"


class GeometryKind(StrEnum):
    NATIVE_COMPONENT = "NATIVE_COMPONENT"
    NUMERICAL_FLOOR_SCALED = "NUMERICAL_FLOOR_SCALED"
    COVARIANCE_WHITENED_SENSITIVITY = "COVARIANCE_WHITENED_SENSITIVITY"


@dataclass(frozen=True, slots=True)
class MetricIdentity(CanonicalRecord):
    """Exact identity of one response distinguishability geometry."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/metric-identity'

    metric_id: str
    geometry_kind: GeometryKind
    coordinate_ids: tuple[str, ...]
    native_units: tuple[str, ...]
    scale_source: str
    probabilistic_model_declared: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.metric_id, field_name="metric_id")
        require_sorted_unique_strings(
            tuple(sorted(self.coordinate_ids)),
            field_name="coordinate_ids",
            allow_empty=False,
        )
        if len(self.coordinate_ids) != len(self.native_units):
            raise ValueError("metric coordinates and native units differ in length")
        if any(not value for value in self.native_units):
            raise ValueError("metric native units must be nonempty")
        validate_nonempty(self.scale_source, field_name="scale_source")
        if self.probabilistic_model_declared:
            raise ValueError("deterministic post-hoc geometry cannot claim a probabilistic model")


@dataclass(frozen=True, slots=True)
class FunctionalBandPoint(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/functional-band-point'

    point_id: str
    receiver_time: Decimal
    coordinate_id: str
    native_unit: str
    mean: Decimal
    lower: Decimal
    upper: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.point_id, field_name="point_id")
        validate_decimal(self.receiver_time, field_name="receiver_time", minimum=Decimal(0))
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        for name, value in (("mean", self.mean), ("lower", self.lower), ("upper", self.upper)):
            validate_decimal(value, field_name=name)
        if self.lower > self.mean or self.mean > self.upper:
            raise ValueError("functional band bounds do not contain the mean")


@dataclass(frozen=True, slots=True)
class FunctionalBandResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/functional-band-result'

    result_id: str
    estimand_id: str
    metric: MetricIdentity
    independent_unit_ids: tuple[str, ...]
    simultaneous_band: bool
    confidence_level: Decimal
    disposition: PosthocDisposition
    points: tuple[FunctionalBandPoint, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_stable_id(self.estimand_id, field_name="estimand_id")
        require_sorted_unique_strings(
            self.independent_unit_ids,
            field_name="independent_unit_ids",
            allow_empty=False,
        )
        validate_decimal(
            self.confidence_level,
            field_name="confidence_level",
            minimum=Decimal(0),
        )
        if self.confidence_level >= 1:
            raise ValueError("confidence level must be below one")
        if not self.points:
            raise ValueError("functional band must contain points")
        point_ids = tuple(value.point_id for value in self.points)
        if tuple(sorted(set(point_ids))) != point_ids:
            raise ValueError("functional band points must have sorted unique IDs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class NumericalEnvelopeRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/numerical-envelope-record'

    envelope_id: str
    primary_view_id: str
    refined_view_id: str
    independent_unit_ids: tuple[str, ...]
    multiplier: Decimal
    paired_complete: bool
    maximum_native_discrepancy: tuple[Decimal, ...]
    coordinate_ids: tuple[str, ...]
    native_units: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("envelope_id", self.envelope_id),
            ("primary_view_id", self.primary_view_id),
            ("refined_view_id", self.refined_view_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.independent_unit_ids,
            field_name="independent_unit_ids",
            allow_empty=False,
        )
        validate_decimal(self.multiplier, field_name="multiplier", minimum=Decimal(1))
        lengths = {
            len(self.maximum_native_discrepancy),
            len(self.coordinate_ids),
            len(self.native_units),
        }
        if len(lengths) != 1 or not self.coordinate_ids:
            raise ValueError("numerical-envelope coordinate fields differ in length")
        if any(value < 0 for value in self.maximum_native_discrepancy):
            raise ValueError("numerical discrepancy cannot be negative")


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TrajectoryPanel:
    """One complete split/numerical-view/state-view trajectory tensor.

    ``values`` is ordered ``[independent_unit, word, receiver_time, coordinate]``.
    The ndarray is an internal computation object and is never serialized as a
    canonical scientific record.
    """

    panel_id: str
    system_id: str
    split_id: str
    numerical_view_id: str
    state_view_id: str
    unit_ids: tuple[str, ...]
    word_ids: tuple[str, ...]
    times: FloatArray
    coordinate_ids: tuple[str, ...]
    native_units: tuple[str, ...]
    values: FloatArray
    source_episode_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        validate_stable_id(self.system_id, field_name="system_id")
        validate_stable_id(self.numerical_view_id, field_name="numerical_view_id")
        validate_stable_id(self.state_view_id, field_name="state_view_id")
        if self.split_id not in {"CALIBRATION", "HELD_OUT"}:
            raise ValueError("trajectory panel split is invalid")
        if tuple(sorted(set(self.unit_ids))) != self.unit_ids or not self.unit_ids:
            raise ValueError("trajectory panel units must be sorted and unique")
        if self.word_ids != WORD_IDS:
            raise ValueError("trajectory panel changes the complete frozen word family")
        if self.times.ndim != 1 or self.times.size == 0:
            raise ValueError("trajectory panel times must be one-dimensional")
        if not np.all(np.isfinite(self.times)) or np.any(np.diff(self.times) <= 0):
            raise ValueError("trajectory panel times must be finite and strictly increasing")
        expected = (
            len(self.unit_ids),
            len(self.word_ids),
            self.times.size,
            len(self.coordinate_ids),
        )
        if self.values.shape != expected or not np.all(np.isfinite(self.values)):
            raise ValueError("trajectory panel values are incomplete or non-finite")
        if len(self.coordinate_ids) != len(self.native_units) or not self.coordinate_ids:
            raise ValueError("trajectory panel coordinates and units differ")
        if len(self.source_episode_sha256s) != len(self.unit_ids) * len(self.word_ids):
            raise ValueError("trajectory panel source identities are incomplete")
        if tuple(sorted(set(self.source_episode_sha256s))) != tuple(
            sorted(self.source_episode_sha256s)
        ):
            raise ValueError("trajectory panel source episode identities are duplicated")

    def word(self, word_id: str) -> FloatArray:
        try:
            index = self.word_ids.index(word_id)
        except ValueError as error:
            raise KeyError(word_id) from error
        return self.values[:, index]


@dataclass(frozen=True, slots=True)
class NumericalEnvelope:
    """Paired numerical discrepancy on identity-relative word responses."""

    absolute: FloatArray  # [word, time, coordinate]
    coordinate_floor: FloatArray  # [time, coordinate]
    record: NumericalEnvelopeRecord


def identity_relative_responses(panel: TrajectoryPanel) -> dict[str, FloatArray]:
    identity = panel.word("identity")
    return {word: panel.word(word) - identity for word in panel.word_ids}


def project_trajectory_panel(
    panel: TrajectoryPanel,
    *,
    state_view_id: str,
    coordinate_ids: tuple[str, ...],
    native_units: tuple[str, ...],
    projection: FloatArray,
) -> TrajectoryPanel:
    """Apply one declared linear receiver gauge to an already verified panel."""

    if projection.shape != (len(coordinate_ids), len(panel.coordinate_ids)):
        raise ValueError("receiver projection shape differs from its declared gauges")
    values = np.einsum("...c,pc->...p", panel.values, projection)
    return TrajectoryPanel(
        panel_id=f"{panel.panel_id}.projected.{state_view_id}",
        system_id=panel.system_id,
        split_id=panel.split_id,
        numerical_view_id=panel.numerical_view_id,
        state_view_id=state_view_id,
        unit_ids=panel.unit_ids,
        word_ids=panel.word_ids,
        times=panel.times.copy(),
        coordinate_ids=coordinate_ids,
        native_units=native_units,
        values=values,
        source_episode_sha256s=panel.source_episode_sha256s,
    )


def subset_trajectory_panel_units(
    panel: TrajectoryPanel, unit_ids: tuple[str, ...]
) -> TrajectoryPanel:
    """Select complete physical units without selecting nested words or rows."""

    if tuple(sorted(set(unit_ids))) != unit_ids or not set(unit_ids) <= set(panel.unit_ids):
        raise ValueError("trajectory panel unit subset is invalid")
    positions = [panel.unit_ids.index(value) for value in unit_ids]
    source_positions: list[int] = []
    for position in positions:
        start = position * len(panel.word_ids)
        source_positions.extend(range(start, start + len(panel.word_ids)))
    return TrajectoryPanel(
        panel_id=f"{panel.panel_id}.units-{len(unit_ids)}",
        system_id=panel.system_id,
        split_id=panel.split_id,
        numerical_view_id=panel.numerical_view_id,
        state_view_id=panel.state_view_id,
        unit_ids=unit_ids,
        word_ids=panel.word_ids,
        times=panel.times.copy(),
        coordinate_ids=panel.coordinate_ids,
        native_units=panel.native_units,
        values=panel.values[positions].copy(),
        source_episode_sha256s=tuple(
            panel.source_episode_sha256s[index] for index in source_positions
        ),
    )


def finite_word_estimands(panel: TrajectoryPanel) -> dict[str, FloatArray]:
    """Return signed, time-resolved finite-word estimands in native coordinates."""

    response = identity_relative_responses(panel)
    raw_order = response["a-then-b"] - response["b-then-a"]
    timing = response["a-early"] + response["b-late"] - response["b-early"] - response["a-late"]
    values = {f"response.{key}": value for key, value in response.items()}
    values.update(
        {
            "simultaneous-additive-defect": (
                response["simultaneous-a-b"] - response["a-early"] - response["b-early"]
            ),
            "raw-order": raw_order,
            "lti-timing-prediction": timing,
            "controlled-order": raw_order - timing,
            "symmetric-composition-defect": (
                0.5 * (response["a-then-b"] + response["b-then-a"])
                - 0.5
                * (
                    response["a-early"]
                    + response["b-late"]
                    + response["b-early"]
                    + response["a-late"]
                )
            ),
            "antisymmetric-composition": 0.5 * raw_order,
            "a-repeat-defect": (response["a-repeat"] - response["a-early"] - response["a-late"]),
            "b-repeat-defect": (response["b-repeat"] - response["b-early"] - response["b-late"]),
        }
    )
    return dict(sorted(values.items()))


def paired_numerical_envelope(
    primary: TrajectoryPanel,
    refined: TrajectoryPanel,
    *,
    multiplier: float = 1.25,
) -> NumericalEnvelope:
    """Construct a conservative paired trajectory envelope without row pooling."""

    if multiplier < 1 or not np.isfinite(multiplier):
        raise ValueError("numerical envelope multiplier must be finite and at least one")
    comparable = (
        primary.system_id == refined.system_id
        and primary.split_id == refined.split_id == "CALIBRATION"
        and primary.state_view_id == refined.state_view_id
        and primary.unit_ids == refined.unit_ids
        and primary.word_ids == refined.word_ids
        and np.array_equal(primary.times, refined.times)
        and primary.coordinate_ids == refined.coordinate_ids
        and primary.native_units == refined.native_units
    )
    if not comparable:
        raise ValueError("numerical views are not a complete paired panel")
    primary_response = identity_relative_responses(primary)
    refined_response = identity_relative_responses(refined)
    discrepancy = np.stack(
        [
            np.max(np.abs(primary_response[word] - refined_response[word]), axis=0)
            for word in primary.word_ids
        ],
        axis=0,
    )
    absolute = discrepancy * multiplier
    coordinate_floor = np.max(absolute, axis=0)
    maxima = np.max(coordinate_floor, axis=0)
    record = NumericalEnvelopeRecord(
        envelope_id=f"envelope.{primary.panel_id}.{refined.numerical_view_id}",
        primary_view_id=primary.numerical_view_id,
        refined_view_id=refined.numerical_view_id,
        independent_unit_ids=primary.unit_ids,
        multiplier=Decimal(str(multiplier)),
        paired_complete=True,
        maximum_native_discrepancy=tuple(Decimal(f"{value:.15g}") for value in maxima),
        coordinate_ids=primary.coordinate_ids,
        native_units=primary.native_units,
    )
    return NumericalEnvelope(absolute=absolute, coordinate_floor=coordinate_floor, record=record)


def bootstrap_functional_band(
    values: FloatArray,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
    simultaneous: bool = True,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Complete-unit mean band over a fixed time/coordinate family.

    When ``simultaneous`` is true, the bootstrap maximum absolute standardized
    deviation supplies one familywise critical value.  Degenerate components
    retain zero-width intervals rather than acquiring artificial noise.
    """

    if values.ndim != 3 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("functional bootstrap requires finite [unit,time,coordinate] values")
    if replicates < 200:
        raise ValueError("functional bootstrap requires at least 200 replicates")
    if not 0.5 < confidence_level < 1:
        raise ValueError("confidence level is invalid")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.shape[0], size=(replicates, values.shape[0]))
    bootstrap_means = np.mean(values[indices], axis=1)
    mean = np.mean(values, axis=0)
    alpha = (1.0 - confidence_level) / 2.0
    if not simultaneous:
        return (
            mean,
            np.quantile(bootstrap_means, alpha, axis=0),
            np.quantile(bootstrap_means, 1.0 - alpha, axis=0),
        )
    standard_error = np.std(bootstrap_means, axis=0, ddof=1)
    safe = np.where(standard_error > 0, standard_error, 1.0)
    standardized = np.abs((bootstrap_means - mean) / safe)
    standardized[:, standard_error == 0] = 0.0
    critical = float(np.quantile(np.max(standardized, axis=(1, 2)), confidence_level))
    half_width = critical * standard_error
    return mean, mean - half_width, mean + half_width


def maximum_leave_one_unit_influence(values: FloatArray) -> FloatArray:
    """Maximum absolute change in the mean after deleting one complete unit."""

    if values.ndim != 3 or values.shape[0] < 3:
        raise ValueError("influence diagnostic requires at least three complete units")
    full = np.mean(values, axis=0)
    leave_one_out = np.stack(
        [np.mean(np.delete(values, index, axis=0), axis=0) for index in range(values.shape[0])]
    )
    return np.asarray(np.max(np.abs(leave_one_out - full), axis=0), dtype=np.float64)


def _safe_floor(floor: FloatArray, panel: TrajectoryPanel) -> FloatArray:
    if floor.shape != (panel.times.size, len(panel.coordinate_ids)):
        raise ValueError("numerical floor differs from panel time/coordinate support")
    state_scale = np.max(np.abs(panel.values), axis=(0, 1, 2))
    epsilon = np.maximum(state_scale * 64 * np.finfo(np.float64).eps, 1e-15)
    return np.asarray(np.maximum(floor, epsilon[None, :]), dtype=np.float64)


def _principal_angle_degrees(first: FloatArray, second: FloatArray) -> float | None:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0 or second_norm == 0:
        return None
    cosine = float(np.clip(abs(np.dot(first, second)) / (first_norm * second_norm), 0, 1))
    return float(np.degrees(np.arccos(cosine)))


def information_geometry_atlas(
    panel: TrajectoryPanel,
    numerical_floor: FloatArray,
) -> dict[str, object]:
    """Finite receiver-pullback ranks, quotients and tangent/normal energies."""

    response = identity_relative_responses(panel)
    floor = _safe_floor(numerical_floor, panel)
    response_matrix = np.stack([np.mean(response[word], axis=0) for word in NONIDENTITY_WORD_IDS])
    rows: list[dict[str, object]] = []
    for time_index, time in enumerate(panel.times):
        native = response_matrix[:, time_index]
        scaled = native / floor[time_index]
        singular = np.linalg.svd(scaled, compute_uv=False)
        perturbation_bound = float(np.sqrt(scaled.size))
        algebraic_rank = int(np.linalg.matrix_rank(scaled))
        certified_rank = int(np.sum(singular > perturbation_bound))
        squared = singular**2
        participation = (
            float(np.sum(squared) ** 2 / np.sum(squared**2)) if np.sum(squared**2) else 0.0
        )

        a = np.mean(response["a-early"][:, time_index], axis=0) / floor[time_index]
        b = np.mean(response["b-early"][:, time_index], axis=0) / floor[time_index]
        controlled = (
            np.mean(finite_word_estimands(panel)["controlled-order"][:, time_index], axis=0)
            / floor[time_index]
        )
        port_matrix = np.column_stack((a, b))
        u, port_singular, _vh = np.linalg.svd(port_matrix, full_matrices=False)
        port_bound = float(np.sqrt(port_matrix.size))
        active = port_singular > port_bound
        if not np.any(active):
            projection_disposition = PosthocDisposition.UNEVALUABLE_PORT_SPAN_UNRESOLVED.value
            parallel = None
            transverse = None
            transverse_fraction = None
        else:
            basis = u[:, active]
            projection = basis @ (basis.T @ controlled)
            residual = controlled - projection
            parallel = float(np.dot(projection, projection))
            transverse = float(np.dot(residual, residual))
            total = parallel + transverse
            transverse_fraction = transverse / total if total else 0.0
            projection_disposition = PosthocDisposition.RESOLVED.value

        equivalence_edges = []
        for left, right in combinations(range(len(NONIDENTITY_WORD_IDS)), 2):
            distance = float(np.linalg.norm(scaled[left] - scaled[right]))
            if distance <= np.sqrt(scaled.shape[1]):
                equivalence_edges.append((NONIDENTITY_WORD_IDS[left], NONIDENTITY_WORD_IDS[right]))
        rows.append(
            {
                "time": float(time),
                "singular_values": singular.tolist(),
                "algebraic_rank": algebraic_rank,
                "floor_certified_rank": certified_rank,
                "perturbation_bound": perturbation_bound,
                "participation_dimension": participation,
                "port_singular_values": port_singular.tolist(),
                "port_perturbation_bound": port_bound,
                "port_angle_degrees": _principal_angle_degrees(a, b),
                "controlled_to_a_angle_degrees": _principal_angle_degrees(controlled, a),
                "controlled_to_b_angle_degrees": _principal_angle_degrees(controlled, b),
                "projection_disposition": projection_disposition,
                "parallel_energy": parallel,
                "transverse_energy": transverse,
                "transverse_fraction": transverse_fraction,
                "word_equivalence_edges": equivalence_edges,
            }
        )

    # Cross-fitted shrinkage-whitened sensitivity; this is explicitly not Fisher information.
    whitened_grams = []
    endpoint = np.stack([response[word][:, -1] for word in NONIDENTITY_WORD_IDS], axis=1)
    for held_out in range(len(panel.unit_ids)):
        training = np.delete(endpoint, held_out, axis=0).reshape(-1, endpoint.shape[-1])
        covariance = np.cov(training, rowvar=False)
        covariance = np.atleast_2d(covariance)
        diagonal = np.diag(np.diag(covariance))
        shrinkage = 0.25 * diagonal + 0.75 * covariance
        scale = max(float(np.trace(shrinkage)) / shrinkage.shape[0], 1e-24)
        inverse = np.linalg.pinv(shrinkage + np.eye(shrinkage.shape[0]) * scale * 1e-8)
        held = endpoint[held_out]
        whitened_grams.append(held @ inverse @ held.T)
    whitened_eigenvalues = np.linalg.eigvalsh(np.mean(whitened_grams, axis=0))[::-1]
    return {
        "metric_label": "receiver-pullback-distinguishability-not-fisher-information",
        "rows": rows,
        "endpoint_cross_fitted_shrinkage_whitened_eigenvalues": whitened_eigenvalues.tolist(),
        "covariance_shrinkage": 0.25,
    }


def _affine_fit(starts: FloatArray, ends: FloatArray, ridge: float) -> FloatArray:
    design = np.column_stack((starts, np.ones(starts.shape[0])))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[-1, -1] = 0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ ends)
    result = np.eye(starts.shape[1] + 1)
    result[:-1, :-1] = coefficients[:-1].T
    result[:-1, -1] = coefficients[-1]
    return result


def _affine_apply(operator: FloatArray, states: FloatArray) -> FloatArray:
    augmented = np.column_stack((states, np.ones(states.shape[0])))
    return (operator @ augmented.T).T[:, :-1]


def _rectangular_affine_fit(starts: FloatArray, ends: FloatArray, ridge: float) -> FloatArray:
    """Fit an affine map when retained-history input and receiver dimensions differ."""

    if starts.ndim != 2 or ends.ndim != 2 or starts.shape[0] != ends.shape[0]:
        raise ValueError("rectangular affine fit requires matched finite rows")
    design = np.column_stack((starts, np.ones(starts.shape[0])))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[-1, -1] = 0
    return np.asarray(
        np.linalg.solve(design.T @ design + penalty, design.T @ ends),
        dtype=np.float64,
    )


def _rectangular_affine_apply(coefficients: FloatArray, states: FloatArray) -> FloatArray:
    design = np.column_stack((states, np.ones(states.shape[0])))
    if design.shape[1] != coefficients.shape[0]:
        raise ValueError("rectangular affine coefficients differ from predictor dimension")
    return design @ coefficients


def cocycle_stationarity_atlas(
    panel: TrajectoryPanel,
    *,
    selected_time_indices: tuple[int, ...],
    ridge: float = 1e-6,
) -> dict[str, object]:
    """Grouped cross-fitted affine identity-flow cocycle/stationarity atlas."""

    if tuple(sorted(set(selected_time_indices))) != selected_time_indices:
        raise ValueError("cocycle time indices must be sorted and unique")
    if selected_time_indices[0] < 0 or selected_time_indices[-1] >= panel.times.size:
        raise ValueError("cocycle time index lies outside the panel")
    identity = panel.word("identity")[:, selected_time_indices]
    scale = np.std(identity.reshape(-1, identity.shape[-1]), axis=0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    scaled = identity / scale
    triples = []
    for first, middle, last in combinations(range(len(selected_time_indices)), 3):
        held_out_defects = []
        direct_errors = []
        composed_errors = []
        for held_out in range(len(panel.unit_ids)):
            train = np.arange(len(panel.unit_ids)) != held_out
            f_middle_first = _affine_fit(scaled[train, first], scaled[train, middle], ridge)
            f_last_middle = _affine_fit(scaled[train, middle], scaled[train, last], ridge)
            f_last_first = _affine_fit(scaled[train, first], scaled[train, last], ridge)
            start_state = scaled[held_out : held_out + 1, first]
            via = _affine_apply(f_last_middle, _affine_apply(f_middle_first, start_state))
            direct = _affine_apply(f_last_first, start_state)
            truth = scaled[held_out : held_out + 1, last]
            held_out_defects.append(float(np.sqrt(np.mean((via - direct) ** 2))))
            direct_errors.append(float(np.sqrt(np.mean((direct - truth) ** 2))))
            composed_errors.append(float(np.sqrt(np.mean((via - truth) ** 2))))
        triples.append(
            {
                "s": float(panel.times[selected_time_indices[first]]),
                "t": float(panel.times[selected_time_indices[middle]]),
                "u": float(panel.times[selected_time_indices[last]]),
                "cocycle_prediction_defect_rms": float(np.mean(held_out_defects)),
                "direct_prediction_rms": float(np.mean(direct_errors)),
                "composed_prediction_rms": float(np.mean(composed_errors)),
            }
        )

    stationarity = []
    count = len(selected_time_indices)
    intervals_by_lag: dict[float, list[tuple[int, int]]] = {}
    for start_index in range(count - 1):
        for end in range(start_index + 1, count):
            lag = float(
                panel.times[selected_time_indices[end]]
                - panel.times[selected_time_indices[start_index]]
            )
            intervals_by_lag.setdefault(lag, []).append((start_index, end))
    for lag, intervals in sorted(intervals_by_lag.items()):
        for (first_anchor, first_end), (second_anchor, second_end) in combinations(intervals, 2):
            matrices = []
            prediction_differences = []
            for held_out in range(len(panel.unit_ids)):
                train = np.arange(len(panel.unit_ids)) != held_out
                first_map = _affine_fit(
                    scaled[train, first_anchor],
                    scaled[train, first_end],
                    ridge,
                )
                second_map = _affine_fit(
                    scaled[train, second_anchor],
                    scaled[train, second_end],
                    ridge,
                )
                matrices.append(
                    float(
                        np.linalg.norm(first_map - second_map, ord="fro") / np.sqrt(first_map.size)
                    )
                )
                first_state = scaled[held_out : held_out + 1, first_anchor]
                prediction_differences.append(
                    float(
                        np.sqrt(
                            np.mean(
                                (
                                    _affine_apply(first_map, first_state)
                                    - _affine_apply(second_map, first_state)
                                )
                                ** 2
                            )
                        )
                    )
                )
            stationarity.append(
                {
                    "lag": lag,
                    "first_anchor": float(panel.times[selected_time_indices[first_anchor]]),
                    "second_anchor": float(panel.times[selected_time_indices[second_anchor]]),
                    "operator_defect_frobenius": float(np.mean(matrices)),
                    "held_out_prediction_difference_rms": float(np.mean(prediction_differences)),
                }
            )

    return {
        "coordinate_scale": scale.tolist(),
        "ridge": ridge,
        "intercept_retained": True,
        "cross_fit_unit": "complete-physical-preparation",
        "triples": triples,
        "stationarity_pairs": stationarity,
    }


def history_sufficiency_analysis(
    panel: TrajectoryPanel,
    numerical_floor: FloatArray,
    *,
    cutoff_indices: tuple[int, ...],
    ridge: float = 1e-5,
) -> list[dict[str, object]]:
    """Compare current-state with one-lag retained-history endpoint prediction."""

    floor = _safe_floor(numerical_floor, panel)[-1]
    rows: list[dict[str, object]] = []
    for cutoff in cutoff_indices:
        if cutoff <= 0 or cutoff >= panel.times.size - 1:
            raise ValueError("history cutoff requires a prior sample and future endpoint")
        current_errors = []
        history_errors = []
        for held_out in range(len(panel.unit_ids)):
            train_units = np.arange(len(panel.unit_ids)) != held_out
            current_train = panel.values[train_units, :, cutoff].reshape(
                -1, len(panel.coordinate_ids)
            )
            prior_train = panel.values[train_units, :, cutoff - 1].reshape(
                -1, len(panel.coordinate_ids)
            )
            target_train = panel.values[train_units, :, -1].reshape(-1, len(panel.coordinate_ids))
            current_map = _affine_fit(current_train, target_train, ridge)
            history_map = _rectangular_affine_fit(
                np.column_stack((current_train, prior_train)), target_train, ridge
            )
            current_test = panel.values[held_out, :, cutoff]
            prior_test = panel.values[held_out, :, cutoff - 1]
            target = panel.values[held_out, :, -1]
            current_prediction = _affine_apply(current_map, current_test)
            history_prediction = _rectangular_affine_apply(
                history_map, np.column_stack((current_test, prior_test))
            )
            current_errors.append(
                float(np.sqrt(np.mean(((current_prediction - target) / floor) ** 2)))
            )
            history_errors.append(
                float(np.sqrt(np.mean(((history_prediction - target) / floor) ** 2)))
            )
        current_rms = float(np.mean(current_errors))
        history_rms = float(np.mean(history_errors))
        rows.append(
            {
                "cutoff_time": float(panel.times[cutoff]),
                "current_state_prediction_rms_floor_units": current_rms,
                "one_lag_history_prediction_rms_floor_units": history_rms,
                "relative_improvement": (
                    (current_rms - history_rms) / current_rms if current_rms else 0.0
                ),
            }
        )
    return rows


def word_confusion_topology(
    panel: TrajectoryPanel,
    numerical_floor: FloatArray,
    *,
    selected_time_indices: tuple[int, ...],
) -> dict[str, object]:
    """Leave-one-preparation-out nearest-centroid word distinguishability."""

    floor = _safe_floor(numerical_floor, panel)[list(selected_time_indices)]
    response = identity_relative_responses(panel)
    features = np.stack(
        [
            response[word][:, selected_time_indices].reshape(len(panel.unit_ids), -1)
            for word in WORD_IDS
        ],
        axis=1,
    )
    feature_floor = np.tile(floor.reshape(-1), 1)
    features = features / feature_floor[None, None, :]
    confusion = np.zeros((len(WORD_IDS), len(WORD_IDS)), dtype=int)
    margins = []
    for held_out in range(len(panel.unit_ids)):
        training = np.arange(len(panel.unit_ids)) != held_out
        centroids = np.mean(features[training], axis=0)
        for true_index in range(len(WORD_IDS)):
            distances = np.linalg.norm(centroids - features[held_out, true_index], axis=1)
            order = np.argsort(distances)
            confusion[true_index, order[0]] += 1
            margins.append(float(distances[order[1]] - distances[order[0]]))
    accuracy = float(np.trace(confusion) / np.sum(confusion))
    return {
        "word_ids": list(WORD_IDS),
        "confusion": confusion.tolist(),
        "accuracy": accuracy,
        "chance_accuracy": 1.0 / len(WORD_IDS),
        "median_nearest_centroid_margin": float(np.median(margins)),
    }


def transient_and_latency_analysis(
    panel: TrajectoryPanel,
    numerical_floor: FloatArray,
) -> dict[str, object]:
    """Time-local materiality, latency, peak and retained-response summaries."""

    response = identity_relative_responses(panel)
    floor = _safe_floor(numerical_floor, panel)
    rows = []
    for word in NONIDENTITY_WORD_IDS:
        mean = np.mean(response[word], axis=0)
        ratio = np.abs(mean) / floor
        for coordinate_index, coordinate_id in enumerate(panel.coordinate_ids):
            resolved = np.flatnonzero(ratio[:, coordinate_index] > 1)
            peak = int(np.argmax(ratio[:, coordinate_index]))
            rows.append(
                {
                    "word_id": word,
                    "coordinate_id": coordinate_id,
                    "native_unit": panel.native_units[coordinate_index],
                    "first_above_floor_time": (
                        float(panel.times[resolved[0]]) if resolved.size else None
                    ),
                    "peak_time": float(panel.times[peak]),
                    "peak_floor_ratio": float(ratio[peak, coordinate_index]),
                    "endpoint_floor_ratio": float(ratio[-1, coordinate_index]),
                    "endpoint_to_peak_ratio": float(
                        abs(mean[-1, coordinate_index])
                        / max(abs(mean[peak, coordinate_index]), 1e-30)
                    ),
                }
            )
    return {"rows": rows}


def finite_relation_map(panel: TrajectoryPanel, numerical_floor: FloatArray) -> dict[str, object]:
    """Observed finite-word relations only; no closure or associativity imputation."""

    estimands = finite_word_estimands(panel)
    floor = _safe_floor(numerical_floor, panel)
    relation_names = (
        "simultaneous-additive-defect",
        "controlled-order",
        "symmetric-composition-defect",
        "a-repeat-defect",
        "b-repeat-defect",
    )
    relations = []
    for name in relation_names:
        mean = np.mean(estimands[name], axis=0)
        ratio = np.abs(mean) / floor
        relations.append(
            {
                "relation_id": name,
                "maximum_floor_ratio": float(np.max(ratio)),
                "endpoint_floor_ratios": ratio[-1].tolist(),
                "supported": bool(np.max(ratio) <= 1),
            }
        )
    return {
        "relations": relations,
        "unobserved_products_imputed": False,
        "associativity_tested": False,
        "inverse_or_undo_tested": False,
        "generator_or_bracket_estimated": False,
    }


def denominator_heterogeneity(
    panel: TrajectoryPanel,
    numerical_floor: FloatArray,
    features: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    """Exploratory rank correlations at the complete preparation level."""

    if set(features) != set(panel.unit_ids):
        raise ValueError("denominator feature rows differ from the panel units")
    estimands = finite_word_estimands(panel)
    endpoints = {
        "port-a": np.abs(identity_relative_responses(panel)["a-early"][:, -1]),
        "port-b": np.abs(identity_relative_responses(panel)["b-early"][:, -1]),
        "controlled-order": np.abs(estimands["controlled-order"][:, -1]),
    }
    floor = _safe_floor(numerical_floor, panel)[-1]
    feature_ids = tuple(sorted(next(iter(features.values()))))
    if any(tuple(sorted(value)) != feature_ids for value in features.values()):
        raise ValueError("denominator feature columns are incomplete")

    def ranks(values: FloatArray) -> FloatArray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty_like(order, dtype=float)
        result[order] = np.arange(values.size, dtype=float)
        for unique in np.unique(values):
            mask = values == unique
            result[mask] = np.mean(result[mask])
        return result

    rows = []
    for feature_id in feature_ids:
        feature_values = np.asarray([features[unit][feature_id] for unit in panel.unit_ids])
        feature_rank = ranks(feature_values)
        for estimand_id, values in endpoints.items():
            for coordinate_index, coordinate_id in enumerate(panel.coordinate_ids):
                response_rank = ranks(values[:, coordinate_index] / floor[coordinate_index])
                if np.std(feature_rank) == 0 or np.std(response_rank) == 0:
                    correlation = 0.0
                else:
                    correlation = float(np.corrcoef(feature_rank, response_rank)[0, 1])
                rows.append(
                    {
                        "feature_id": feature_id,
                        "estimand_id": estimand_id,
                        "coordinate_id": coordinate_id,
                        "spearman_rank_correlation": correlation,
                    }
                )
    return {"rows": rows, "independent_unit_count": len(panel.unit_ids)}


def split_recurrence(
    development: TrajectoryPanel,
    evaluation: TrajectoryPanel,
    development_floor: FloatArray,
) -> dict[str, object]:
    """Compare outcome-visible split mean paths without treating evaluation as fresh."""

    if (
        development.system_id != evaluation.system_id
        or development.state_view_id != evaluation.state_view_id
        or development.word_ids != evaluation.word_ids
        or not np.array_equal(development.times, evaluation.times)
    ):
        raise ValueError("split recurrence panels are incomparable")
    floor = _safe_floor(development_floor, development)
    dev = identity_relative_responses(development)
    evaluation_response = identity_relative_responses(evaluation)
    rows = []
    for word in NONIDENTITY_WORD_IDS:
        left = (np.mean(dev[word], axis=0) / floor).reshape(-1)
        right = (np.mean(evaluation_response[word], axis=0) / floor).reshape(-1)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        rows.append(
            {
                "word_id": word,
                "cosine_similarity": float(np.dot(left, right) / denominator)
                if denominator
                else None,
                "normalized_rms_difference": float(np.sqrt(np.mean((left - right) ** 2))),
            }
        )
    return {
        "rows": rows,
        "evaluation_is_fresh_confirmation": False,
        "outcome_visible": True,
    }


def gauge_naturality(
    full_panel: TrajectoryPanel,
    primary_panel: TrajectoryPanel,
    projection: FloatArray,
) -> dict[str, object]:
    """Check that projection and finite-word differencing commute."""

    if projection.shape != (len(primary_panel.coordinate_ids), len(full_panel.coordinate_ids)):
        raise ValueError("gauge projection shape differs from the state views")
    if (
        full_panel.unit_ids != primary_panel.unit_ids
        or full_panel.word_ids != primary_panel.word_ids
        or not np.array_equal(full_panel.times, primary_panel.times)
    ):
        raise ValueError("gauge panels are not nested observations of the same episodes")
    projected_values = np.einsum("...c,pc->...p", full_panel.values, projection)
    value_defect = projected_values - primary_panel.values
    full_estimands = finite_word_estimands(full_panel)
    primary_estimands = finite_word_estimands(primary_panel)
    estimand_defects = {
        name: float(
            np.max(np.abs(np.einsum("...c,pc->...p", full_estimands[name], projection) - value))
        )
        for name, value in primary_estimands.items()
    }
    return {
        "maximum_value_projection_defect": float(np.max(np.abs(value_defect))),
        "maximum_estimand_naturality_defect": max(estimand_defects.values()),
        "estimand_defects": estimand_defects,
        "projection_exact": bool(np.max(np.abs(value_defect)) <= 1e-13),
    }


def operator_spectrum_sensitivity(
    panel: TrajectoryPanel,
    *,
    selected_time_indices: tuple[int, ...],
    ridge: float = 1e-6,
) -> dict[str, object]:
    """Anchor-local affine linear-part spectra, labelled as sensitivity only."""

    identity = panel.word("identity")[:, selected_time_indices]
    scale = np.std(identity.reshape(-1, identity.shape[-1]), axis=0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    identity = identity / scale
    rows = []
    for start in range(len(selected_time_indices) - 1):
        for end in range(start + 1, len(selected_time_indices)):
            operator = _affine_fit(identity[:, start], identity[:, end], ridge)
            eigenvalues = np.linalg.eigvals(operator[:-1, :-1])
            rows.append(
                {
                    "start": float(panel.times[selected_time_indices[start]]),
                    "end": float(panel.times[selected_time_indices[end]]),
                    "lag": float(
                        panel.times[selected_time_indices[end]]
                        - panel.times[selected_time_indices[start]]
                    ),
                    "spectral_radius": float(np.max(np.abs(eigenvalues))),
                    "eigenvalue_real": np.real(eigenvalues).tolist(),
                    "eigenvalue_imaginary": np.imag(eigenvalues).tolist(),
                }
            )
    return {
        "rows": rows,
        "interpretation": "finite-sample-affine-sensitivity-not-physical-generator-spectrum",
    }


def affine_model_adequacy(
    panel: TrajectoryPanel,
    *,
    selected_time_indices: tuple[int, ...],
    ridge: float = 1e-6,
) -> dict[str, object]:
    """Compare persistence, stationary-lag and anchor-local affine identity maps."""

    identity = panel.word("identity")[:, selected_time_indices]
    scale = np.std(identity.reshape(-1, identity.shape[-1]), axis=0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    identity = identity / scale
    intervals_by_lag: dict[float, list[tuple[int, int]]] = {}
    for start in range(len(selected_time_indices) - 1):
        for end in range(start + 1, len(selected_time_indices)):
            lag = float(
                panel.times[selected_time_indices[end]] - panel.times[selected_time_indices[start]]
            )
            intervals_by_lag.setdefault(lag, []).append((start, end))
    rows = []
    for lag, intervals in sorted(intervals_by_lag.items()):
        for start, end in intervals:
            errors: dict[str, list[float]] = {
                "persistence": [],
                "stationary-lag-affine": [],
                "anchor-local-affine": [],
            }
            for held_out in range(len(panel.unit_ids)):
                train = np.arange(len(panel.unit_ids)) != held_out
                anchor_map = _affine_fit(identity[train, start], identity[train, end], ridge)
                pooled_starts = np.concatenate(
                    [identity[train, left] for left, _right in intervals], axis=0
                )
                pooled_ends = np.concatenate(
                    [identity[train, right] for _left, right in intervals], axis=0
                )
                stationary_map = _affine_fit(pooled_starts, pooled_ends, ridge)
                source = identity[held_out : held_out + 1, start]
                truth = identity[held_out : held_out + 1, end]
                predictions = {
                    "persistence": source,
                    "stationary-lag-affine": _affine_apply(stationary_map, source),
                    "anchor-local-affine": _affine_apply(anchor_map, source),
                }
                for model_id, prediction in predictions.items():
                    errors[model_id].append(float(np.sqrt(np.mean((prediction - truth) ** 2))))
            rows.append(
                {
                    "lag": lag,
                    "anchor": float(panel.times[selected_time_indices[start]]),
                    **{f"{key}_rms": float(np.mean(value)) for key, value in errors.items()},
                }
            )
    aggregate = {
        model: float(np.mean([row[f"{model}_rms"] for row in rows]))
        for model in ("persistence", "stationary-lag-affine", "anchor-local-affine")
    }
    return {"rows": rows, "aggregate_normalized_rms": aggregate, "intercept_retained": True}


def precision_design_value(
    panel: TrajectoryPanel,
    numerical_floor: FloatArray,
    estimand_bands: Mapping[str, tuple[FloatArray, FloatArray, FloatArray]],
) -> dict[str, object]:
    """Compare numerical envelope and preparation uncertainty for next-design value."""

    floor = _safe_floor(numerical_floor, panel)
    response = identity_relative_responses(panel)
    peak_ratios = {
        word: float(np.max(np.abs(np.mean(values, axis=0)) / floor))
        for word, values in response.items()
        if word != "identity"
    }
    uncertainty = {}
    for estimand, (mean, lower, upper) in estimand_bands.items():
        bootstrap_half_width = np.maximum(mean - lower, upper - mean)
        uncertainty[estimand] = {
            "maximum_bootstrap_to_numerical_ratio": float(np.max(bootstrap_half_width / floor)),
            "median_bootstrap_to_numerical_ratio": float(np.median(bootstrap_half_width / floor)),
        }
    median_peak = float(np.median(list(peak_ratios.values())))
    if median_peak <= 1:
        priority = "IMPROVE_NUMERICAL_PRECISION_OR_ACTION_DESIGN_BEFORE_ADDING_UNITS"
    else:
        ratios = [value["median_bootstrap_to_numerical_ratio"] for value in uncertainty.values()]
        priority = (
            "ADD_INDEPENDENT_PREPARATIONS"
            if ratios and float(np.median(ratios)) > 1
            else "EXTEND_ACTION_TIME_DOSE_DESIGN"
        )
    return {
        "peak_response_to_floor_by_word": peak_ratios,
        "uncertainty": uncertainty,
        "next_design_priority": priority,
    }


def floor_disposition(
    mean: FloatArray, lower: FloatArray, upper: FloatArray, floor: FloatArray
) -> PosthocDisposition:
    """Classify a complete functional family against its numerical envelope."""

    if mean.shape != lower.shape or mean.shape != upper.shape or mean.shape != floor.shape:
        raise ValueError("functional disposition arrays differ in shape")
    if np.all(np.maximum(np.abs(lower), np.abs(upper)) <= floor):
        return PosthocDisposition.BELOW_NUMERICAL_RESOLUTION
    minimum_magnitude = np.where(
        (lower <= 0) & (upper >= 0),
        0.0,
        np.minimum(np.abs(lower), np.abs(upper)),
    )
    if np.any(minimum_magnitude > floor):
        return PosthocDisposition.RESOLVED
    return PosthocDisposition.ANNULAR


__all__ = [
    "FunctionalBandPoint",
    "FunctionalBandResult",
    "GeometryKind",
    "MetricIdentity",
    "NONIDENTITY_WORD_IDS",
    "NumericalEnvelope",
    "NumericalEnvelopeRecord",
    "PosthocDisposition",
    "TrajectoryPanel",
    "WORD_IDS",
    "bootstrap_functional_band",
    "affine_model_adequacy",
    "cocycle_stationarity_atlas",
    "denominator_heterogeneity",
    "finite_relation_map",
    "finite_word_estimands",
    "floor_disposition",
    "gauge_naturality",
    "history_sufficiency_analysis",
    "identity_relative_responses",
    "information_geometry_atlas",
    "maximum_leave_one_unit_influence",
    "operator_spectrum_sensitivity",
    "paired_numerical_envelope",
    "precision_design_value",
    "project_trajectory_panel",
    "split_recurrence",
    "transient_and_latency_analysis",
    "subset_trajectory_panel_units",
    "word_confusion_topology",
]
