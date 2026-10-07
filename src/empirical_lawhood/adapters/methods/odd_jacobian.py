"""Odd state-dependent relational Jacobian identification.

This substrate-neutral method treats a physical preparation as the fitting
unit and comparator/plus/minus branches as nested observations.  It constructs
central-difference slopes, fits state-independent and state-dependent slope
models from development preparations only, and predicts response through a
linear native-action contraction.  Consequently every prediction is odd in
action and is exactly zero at zero action.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, Final

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling, inherited_visibility
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)

from .contracts import DataSplit


ODD_RELATIONAL_JACOBIAN_METHOD_KEY: Final = "law.odd-relational-jacobian"
ODD_RELATIONAL_JACOBIAN_METHOD_VERSION: Final = "1.0.0"


class OddJacobianModelKind(StrEnum):
    STATE_INDEPENDENT = "STATE_INDEPENDENT"
    STATE_DEPENDENT = "STATE_DEPENDENT"


@dataclass(frozen=True, slots=True)
class OddJacobianCoordinate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/odd-jacobian-coordinate'

    coordinate_id: str
    native_unit: str

    def __post_init__(self) -> None:
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        validate_nonempty(self.native_unit, field_name="native_unit")


@dataclass(frozen=True, slots=True)
class OddJacobianActionCoordinate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/odd-jacobian-action-coordinate'

    port_id: str
    native_unit: str
    central_difference_amplitude: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.port_id, field_name="port_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_decimal(
            self.central_difference_amplitude,
            field_name="central_difference_amplitude",
            minimum=Decimal(0),
        )
        if self.central_difference_amplitude == 0:
            raise ValueError("odd Jacobian action amplitude must be positive")


@dataclass(frozen=True, slots=True)
class OddJacobianReceiverCoordinate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/odd-jacobian-receiver-coordinate'

    coordinate_id: str
    family_id: str
    native_unit: str

    def __post_init__(self) -> None:
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        validate_stable_id(self.family_id, field_name="family_id")
        validate_nonempty(self.native_unit, field_name="native_unit")


@dataclass(frozen=True, slots=True)
class OddJacobianBranchObservation(CanonicalRecord):
    """One comparator or signed branch nested under a preparation/horizon."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/odd-jacobian-branch-observation'

    observation_id: str
    physical_preparation_id: str
    split: DataSplit
    horizon_s: int
    causal_cutoff_s: int
    state_clock_s: int
    receiver_clock_s: int
    port_id: str | None
    signed_action: Decimal
    action_unit: str
    state_values: tuple[NamedDecimal, ...]
    receiver_values: tuple[NamedDecimal, ...]
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.observation_id, field_name="observation_id")
        validate_stable_id(
            self.physical_preparation_id,
            field_name="physical_preparation_id",
        )
        if self.horizon_s <= 0:
            raise ValueError("odd Jacobian horizon must be positive")
        if self.causal_cutoff_s < 0 or self.receiver_clock_s <= self.causal_cutoff_s:
            raise ValueError("odd Jacobian receiver clock must follow the causal cutoff")
        if self.state_clock_s < 0 or self.state_clock_s > self.causal_cutoff_s:
            raise ValueError("odd Jacobian state must be observed by the causal cutoff")
        validate_decimal(self.signed_action, field_name="signed_action")
        validate_nonempty(self.action_unit, field_name="action_unit")
        if self.port_id is None:
            if self.signed_action != 0 or self.action_unit != "1":
                raise ValueError("odd Jacobian comparator must have zero dimensionless action")
        else:
            validate_stable_id(self.port_id, field_name="port_id")
            if self.signed_action == 0:
                raise ValueError("odd Jacobian signed branch cannot have zero action")
        for field_name, values in (
            ("state_values", self.state_values),
            ("receiver_values", self.receiver_values),
        ):
            require_sorted_unique_ids(values, attribute="value_id", field_name=field_name)
            if not values:
                raise ValueError(f"{field_name} must not be empty")
        require_sorted_unique_strings(self.tags, field_name="tags")


@dataclass(frozen=True, slots=True)
class OddJacobianDataset(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/odd-jacobian-dataset'

    dataset_id: str
    system: ObjectIdentity
    information_cutoff_id: str
    observations: tuple[OddJacobianBranchObservation, ...]
    evidence_artifacts: tuple[ArtifactIdentity, ...]
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    planned_evaluation_preparation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.dataset_id, field_name="dataset_id")
        validate_stable_id(self.information_cutoff_id, field_name="information_cutoff_id")
        if not isinstance(self.system, ObjectIdentity):
            raise ValueError("odd Jacobian dataset requires an exact system identity")
        require_sorted_unique_ids(
            self.observations,
            attribute="observation_id",
            field_name="observations",
        )
        if not self.observations:
            raise ValueError("odd Jacobian dataset requires branch observations")
        require_sorted_unique_ids(
            self.evidence_artifacts,
            attribute="artifact_id",
            field_name="evidence_artifacts",
        )
        if not self.evidence_artifacts:
            raise ValueError("odd Jacobian dataset requires immutable evidence identities")
        require_sorted_unique_strings(
            self.planned_evaluation_preparation_ids,
            field_name="planned_evaluation_preparation_ids",
        )
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("odd Jacobian dataset visibility cannot be lowered")


@dataclass(frozen=True, slots=True)
class OddRelationalJacobianConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/odd-relational-jacobian-config'

    config_id: str
    method_key: str
    method_version: str
    state_coordinates: tuple[OddJacobianCoordinate, ...]
    action_coordinates: tuple[OddJacobianActionCoordinate, ...]
    receiver_coordinates: tuple[OddJacobianReceiverCoordinate, ...]
    ridge_candidates: tuple[Decimal, ...]
    maximum_design_condition_number: Decimal
    minimum_scale: Decimal
    minimum_active_state_coordinates: int
    regularization_selection: str
    state_support_selection: str
    development_scaling_only: bool
    complete_signs_required: bool
    exact_zero_action: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            self.method_key != ODD_RELATIONAL_JACOBIAN_METHOD_KEY
            or self.method_version != ODD_RELATIONAL_JACOBIAN_METHOD_VERSION
        ):
            raise ValueError("odd Jacobian config selects another method")
        require_sorted_unique_ids(
            self.state_coordinates,
            attribute="coordinate_id",
            field_name="state_coordinates",
        )
        require_sorted_unique_ids(
            self.action_coordinates,
            attribute="port_id",
            field_name="action_coordinates",
        )
        require_sorted_unique_ids(
            self.receiver_coordinates,
            attribute="coordinate_id",
            field_name="receiver_coordinates",
        )
        if (
            not self.state_coordinates
            or not self.action_coordinates
            or not self.receiver_coordinates
        ):
            raise ValueError("odd Jacobian config requires state, action and receiver coordinates")
        if tuple(sorted(set(self.ridge_candidates))) != self.ridge_candidates:
            raise ValueError("ridge candidates must be sorted and unique")
        if not self.ridge_candidates:
            raise ValueError("odd Jacobian config requires a fixed ridge candidate grid")
        for value in self.ridge_candidates:
            validate_decimal(value, field_name="ridge_candidate", minimum=Decimal(0))
        validate_decimal(
            self.maximum_design_condition_number,
            field_name="maximum_design_condition_number",
            minimum=Decimal(1),
        )
        validate_decimal(self.minimum_scale, field_name="minimum_scale", minimum=Decimal(0))
        if self.minimum_scale == 0:
            raise ValueError("odd Jacobian minimum scale must be positive")
        if not 1 <= self.minimum_active_state_coordinates <= len(self.state_coordinates):
            raise ValueError("odd Jacobian active-state requirement is outside the frozen state")
        if self.regularization_selection != "leave-one-preparation-out-family-balanced":
            raise ValueError("odd Jacobian regularization selection is not registered")
        if self.state_support_selection != "development-scale-rank-condition-greedy":
            raise ValueError("odd Jacobian state support selection is not registered")
        if not (
            self.development_scaling_only
            and self.complete_signs_required
            and self.exact_zero_action
        ):
            raise ValueError("odd Jacobian scientific invariants cannot be disabled")


@dataclass(frozen=True, slots=True)
class OddJacobianCoefficient(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/odd-jacobian-coefficient'

    coefficient_id: str
    receiver_coordinate_id: str
    term_id: str
    value: Decimal
    native_unit: str

    def __post_init__(self) -> None:
        validate_stable_id(self.coefficient_id, field_name="coefficient_id")
        validate_stable_id(self.receiver_coordinate_id, field_name="receiver_coordinate_id")
        validate_stable_id(self.term_id, field_name="term_id")
        validate_decimal(self.value, field_name="value")
        validate_nonempty(self.native_unit, field_name="native_unit")


@dataclass(frozen=True, slots=True)
class OddJacobianModel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/odd-jacobian-model'

    model_id: str
    method_key: str
    method_version: str
    model_kind: OddJacobianModelKind
    port_id: str
    action_unit: str
    horizon_s: int
    state_centres: tuple[NamedDecimal, ...]
    state_scales: tuple[NamedDecimal, ...]
    receiver_coordinate_ids: tuple[str, ...]
    family_scales: tuple[NamedDecimal, ...]
    coefficients: tuple[OddJacobianCoefficient, ...]
    ridge_alpha: Decimal
    design_rank: int
    design_condition_number: Decimal
    development_preparation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.model_id, field_name="model_id")
        if (
            self.method_key != ODD_RELATIONAL_JACOBIAN_METHOD_KEY
            or self.method_version != ODD_RELATIONAL_JACOBIAN_METHOD_VERSION
        ):
            raise ValueError("odd Jacobian model has another method identity")
        validate_stable_id(self.port_id, field_name="port_id")
        validate_nonempty(self.action_unit, field_name="action_unit")
        if self.horizon_s <= 0:
            raise ValueError("odd Jacobian model horizon must be positive")
        require_sorted_unique_ids(
            self.state_centres, attribute="value_id", field_name="state_centres"
        )
        require_sorted_unique_ids(
            self.state_scales, attribute="value_id", field_name="state_scales"
        )
        if tuple(value.value_id for value in self.state_centres) != tuple(
            value.value_id for value in self.state_scales
        ):
            raise ValueError("odd Jacobian state scaling coordinates differ")
        require_sorted_unique_strings(
            self.receiver_coordinate_ids,
            field_name="receiver_coordinate_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.family_scales, attribute="value_id", field_name="family_scales"
        )
        require_sorted_unique_ids(
            self.coefficients, attribute="coefficient_id", field_name="coefficients"
        )
        validate_decimal(self.ridge_alpha, field_name="ridge_alpha", minimum=Decimal(0))
        validate_decimal(
            self.design_condition_number,
            field_name="design_condition_number",
            minimum=Decimal(1),
        )
        expected_rank = 1 + (
            len(self.state_centres)
            if self.model_kind is OddJacobianModelKind.STATE_DEPENDENT
            else 0
        )
        if self.design_rank != expected_rank:
            raise ValueError("odd Jacobian model design is not full declared rank")
        require_sorted_unique_strings(
            self.development_preparation_ids,
            field_name="development_preparation_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class OddJacobianFit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/odd-jacobian-fit'

    fit_id: str
    dataset: ObjectIdentity
    config: ObjectIdentity
    models: tuple[OddJacobianModel, ...]
    development_preparation_ids: tuple[str, ...]
    evaluation_preparation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.fit_id, field_name="fit_id")
        if not isinstance(self.dataset, ObjectIdentity) or not isinstance(
            self.config, ObjectIdentity
        ):
            raise ValueError("odd Jacobian fit requires exact dataset and config identities")
        require_sorted_unique_ids(self.models, attribute="model_id", field_name="models")
        for field_name, values in (
            ("development_preparation_ids", self.development_preparation_ids),
            ("evaluation_preparation_ids", self.evaluation_preparation_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        if set(self.development_preparation_ids) & set(self.evaluation_preparation_ids):
            raise ValueError("odd Jacobian development/evaluation preparations overlap")


@dataclass(frozen=True, slots=True)
class _Triplet:
    preparation_id: str
    split: DataSplit
    horizon_s: int
    port_id: str
    amplitude: float
    state: npt.NDArray[np.float64]
    plus: npt.NDArray[np.float64]
    minus: npt.NDArray[np.float64]
    comparator: npt.NDArray[np.float64]

    @property
    def slope(self) -> npt.NDArray[np.float64]:
        return (self.plus - self.minus) / (2.0 * self.amplitude)

    @property
    def even(self) -> npt.NDArray[np.float64]:
        return (self.plus + self.minus - 2.0 * self.comparator) / 2.0


def _values(values: tuple[NamedDecimal, ...], expected: tuple[str, ...]) -> np.ndarray:
    by_id = {value.value_id: float(value.value) for value in values}
    if tuple(sorted(by_id)) != expected:
        raise ValueError("odd Jacobian observation coordinates differ from the frozen config")
    return np.asarray(tuple(by_id[value_id] for value_id in expected), dtype=np.float64)


def _selected_values(values: tuple[NamedDecimal, ...], selected: tuple[str, ...]) -> np.ndarray:
    by_id = {value.value_id: float(value.value) for value in values}
    if not set(selected).issubset(by_id):
        raise ValueError("odd Jacobian observation lacks a selected state coordinate")
    return np.asarray(tuple(by_id[value_id] for value_id in selected), dtype=np.float64)


def _validate_units(
    observations: tuple[OddJacobianBranchObservation, ...],
    config: OddRelationalJacobianConfig,
) -> None:
    state_units = {value.coordinate_id: value.native_unit for value in config.state_coordinates}
    receiver_units = {
        value.coordinate_id: value.native_unit for value in config.receiver_coordinates
    }
    action_units = {value.port_id: value.native_unit for value in config.action_coordinates}
    for observation in observations:
        if {value.value_id: value.unit for value in observation.state_values} != state_units:
            raise ValueError("odd Jacobian state unit mismatch")
        if {value.value_id: value.unit for value in observation.receiver_values} != receiver_units:
            raise ValueError("odd Jacobian receiver unit mismatch")
        if observation.port_id is not None and observation.action_unit != action_units.get(
            observation.port_id
        ):
            raise ValueError("odd Jacobian action unit mismatch")


def _triplets(
    dataset: OddJacobianDataset,
    config: OddRelationalJacobianConfig,
) -> tuple[_Triplet, ...]:
    _validate_units(dataset.observations, config)
    state_ids = tuple(value.coordinate_id for value in config.state_coordinates)
    receiver_ids = tuple(value.coordinate_id for value in config.receiver_coordinates)
    action_by_id = {value.port_id: value for value in config.action_coordinates}
    groups: dict[
        tuple[str, DataSplit, int],
        list[OddJacobianBranchObservation],
    ] = {}
    for observation in dataset.observations:
        groups.setdefault(
            (observation.physical_preparation_id, observation.split, observation.horizon_s),
            [],
        ).append(observation)
    values: list[_Triplet] = []
    for (preparation_id, split, horizon_s), rows in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1].value, item[0][2])
    ):
        comparators = tuple(row for row in rows if row.port_id is None)
        if len(comparators) != 1:
            raise ValueError("odd Jacobian cell/horizon requires exactly one comparator")
        comparator = comparators[0]
        reference_state = comparator.state_values
        reference_state_clock = comparator.state_clock_s
        reference_cutoff = comparator.causal_cutoff_s
        reference_clock = comparator.receiver_clock_s
        for row in rows:
            if (
                row.state_values != reference_state
                or row.state_clock_s != reference_state_clock
                or row.causal_cutoff_s != reference_cutoff
                or row.receiver_clock_s != reference_clock
            ):
                raise ValueError("odd Jacobian branches do not share one causal state/clock")
        for port_id, action in sorted(action_by_id.items()):
            port_rows = tuple(row for row in rows if row.port_id == port_id)
            plus = tuple(row for row in port_rows if row.signed_action > 0)
            minus = tuple(row for row in port_rows if row.signed_action < 0)
            if len(plus) != 1 or len(minus) != 1 or len(port_rows) != 2:
                raise ValueError("odd Jacobian cell/port/horizon requires complete +/- signs")
            if (
                plus[0].signed_action != action.central_difference_amplitude
                or minus[0].signed_action != -action.central_difference_amplitude
            ):
                raise ValueError("odd Jacobian signed amplitude differs from the frozen chart")
            values.append(
                _Triplet(
                    preparation_id=preparation_id,
                    split=split,
                    horizon_s=horizon_s,
                    port_id=port_id,
                    amplitude=float(action.central_difference_amplitude),
                    state=_values(reference_state, state_ids),
                    plus=_values(plus[0].receiver_values, receiver_ids),
                    minus=_values(minus[0].receiver_values, receiver_ids),
                    comparator=_values(comparator.receiver_values, receiver_ids),
                )
            )
        expected = 1 + 2 * len(action_by_id)
        if len(rows) != expected:
            raise ValueError("odd Jacobian cell/horizon contains extra branches")
    development = {value.preparation_id for value in values if value.split is DataSplit.CALIBRATION}
    observed_evaluation = {
        value.preparation_id for value in values if value.split is DataSplit.HELD_OUT
    }
    evaluation = (
        set(dataset.planned_evaluation_preparation_ids)
        if dataset.planned_evaluation_preparation_ids
        else observed_evaluation
    )
    if (
        not development
        or not evaluation
        or development & evaluation
        or not observed_evaluation.issubset(evaluation)
    ):
        raise ValueError("odd Jacobian requires disjoint development/evaluation preparations")
    return tuple(values)


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("odd Jacobian method produced a non-finite value")
    return Decimal(f"{value:.17g}")


def _ridge(
    design: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    matrix = design.T @ design + alpha * penalty
    try:
        return np.linalg.solve(matrix, design.T @ target)
    except np.linalg.LinAlgError as exc:
        raise ValueError("odd Jacobian regularized design is singular") from exc


def _family_indices(config: OddRelationalJacobianConfig) -> dict[str, np.ndarray]:
    families: dict[str, list[int]] = {}
    for index, coordinate in enumerate(config.receiver_coordinates):
        families.setdefault(coordinate.family_id, []).append(index)
    return {
        family_id: np.asarray(indices, dtype=np.int64)
        for family_id, indices in sorted(families.items())
    }


def _family_scales(
    triplets: tuple[_Triplet, ...],
    config: OddRelationalJacobianConfig,
    horizon_s: int,
    port_id: str,
) -> dict[str, float]:
    families = _family_indices(config)
    slopes = np.asarray(
        tuple(
            value.slope
            for value in triplets
            if value.split is DataSplit.CALIBRATION
            and value.horizon_s == horizon_s
            and value.port_id == port_id
        ),
        dtype=np.float64,
    )
    if not slopes.size:
        raise ValueError("odd Jacobian horizon has no development slopes")
    minimum = float(config.minimum_scale)
    return {
        family_id: max(float(np.sqrt(np.mean(np.square(slopes[:, indices])))), minimum)
        for family_id, indices in families.items()
    }


def _balanced_error(
    residual: np.ndarray,
    family_scales: dict[str, float],
    config: OddRelationalJacobianConfig,
) -> float:
    return float(
        np.mean(
            tuple(
                np.sqrt(np.mean(np.square(residual[:, indices]))) / family_scales[family_id]
                for family_id, indices in _family_indices(config).items()
            )
        )
    )


def _select_alpha(
    design: np.ndarray,
    target: np.ndarray,
    preparation_ids: tuple[str, ...],
    candidates: tuple[Decimal, ...],
    family_scales: dict[str, float],
    config: OddRelationalJacobianConfig,
) -> float:
    if len(preparation_ids) != design.shape[0]:
        raise ValueError("odd Jacobian rows are not one per development preparation")
    scores: list[tuple[float, float]] = []
    for candidate in candidates:
        alpha = float(candidate)
        residuals = []
        evaluable = True
        for index in range(design.shape[0]):
            keep = np.arange(design.shape[0]) != index
            try:
                coefficients = _ridge(design[keep], target[keep], alpha)
            except ValueError:
                evaluable = False
                break
            residuals.append(target[index] - design[index] @ coefficients)
        if evaluable:
            score = _balanced_error(np.asarray(residuals, dtype=np.float64), family_scales, config)
            scores.append((score, alpha))
    if not scores:
        raise ValueError("odd Jacobian ridge grid has no evaluable candidate")
    return min(scores, key=lambda value: (value[0], value[1]))[1]


def fit_odd_relational_jacobian(
    dataset: OddJacobianDataset,
    config: OddRelationalJacobianConfig,
) -> OddJacobianFit:
    "Fit the state-independent and state-dependent models using development preparations only."

    triplets = _triplets(dataset, config)
    receiver_ids = tuple(value.coordinate_id for value in config.receiver_coordinates)
    receiver_units = {
        value.coordinate_id: value.native_unit for value in config.receiver_coordinates
    }
    action_units = {value.port_id: value.native_unit for value in config.action_coordinates}
    development_ids = tuple(
        sorted({value.preparation_id for value in triplets if value.split is DataSplit.CALIBRATION})
    )
    evaluation_ids = (
        dataset.planned_evaluation_preparation_ids
        if dataset.planned_evaluation_preparation_ids
        else tuple(
            sorted(
                {value.preparation_id for value in triplets if value.split is DataSplit.HELD_OUT}
            )
        )
    )
    exemplar_by_id = {
        value.preparation_id: value for value in triplets if value.split is DataSplit.CALIBRATION
    }
    raw_state_all = np.asarray(
        tuple(exemplar_by_id[value].state for value in development_ids), dtype=np.float64
    )
    centres_all = np.mean(raw_state_all, axis=0)
    scales_all = np.std(raw_state_all, axis=0, ddof=0)
    eligible = tuple(
        index for index, scale in enumerate(scales_all) if scale >= float(config.minimum_scale)
    )
    if not eligible:
        raise ValueError("odd Jacobian development state has no supported coordinate")
    candidate = (raw_state_all[:, eligible] - centres_all[list(eligible)]) / scales_all[
        list(eligible)
    ]
    selected_positions: list[int] = []
    support_design = np.ones((len(development_ids), 1), dtype=np.float64)
    current_rank = 1
    for position in range(candidate.shape[1]):
        proposed = np.column_stack((support_design, candidate[:, position]))
        proposed_rank = int(np.linalg.matrix_rank(proposed))
        proposed_condition = float(np.linalg.cond(proposed))
        if proposed_rank == current_rank + 1 and proposed_condition <= float(
            config.maximum_design_condition_number
        ):
            selected_positions.append(position)
            support_design = proposed
            current_rank = proposed_rank
    if len(selected_positions) < config.minimum_active_state_coordinates:
        raise ValueError(
            "odd Jacobian development state support is insufficient or ill-conditioned"
        )
    selected_indices = tuple(eligible[position] for position in selected_positions)
    active_state_coordinates = tuple(config.state_coordinates[index] for index in selected_indices)
    centres = centres_all[list(selected_indices)]
    scales = scales_all[list(selected_indices)]
    standardized = (raw_state_all[:, selected_indices] - centres) / scales
    full_design = np.column_stack((np.ones(len(development_ids)), standardized))
    full_rank = int(np.linalg.matrix_rank(full_design))
    condition = float(np.linalg.cond(full_design))
    if full_rank != full_design.shape[1]:
        raise ValueError("odd Jacobian development design is rank deficient")
    if condition > float(config.maximum_design_condition_number):
        raise ValueError("odd Jacobian development design is ill-conditioned")
    state_centres = tuple(
        NamedDecimal(
            value_id=value.coordinate_id, value=_decimal(centres[index]), unit=value.native_unit
        )
        for index, value in enumerate(active_state_coordinates)
    )
    state_scales = tuple(
        NamedDecimal(
            value_id=value.coordinate_id, value=_decimal(scales[index]), unit=value.native_unit
        )
        for index, value in enumerate(active_state_coordinates)
    )
    horizons = tuple(sorted({value.horizon_s for value in triplets}))
    models: list[OddJacobianModel] = []
    for horizon_s in horizons:
        for action in config.action_coordinates:
            family_scales = _family_scales(triplets, config, horizon_s, action.port_id)
            family_scale_records = tuple(
                NamedDecimal(
                    value_id=family_id,
                    value=_decimal(scale),
                    unit=f"1/({action.native_unit})",
                )
                for family_id, scale in family_scales.items()
            )
            by_preparation = {
                value.preparation_id: value
                for value in triplets
                if value.split is DataSplit.CALIBRATION
                and value.horizon_s == horizon_s
                and value.port_id == action.port_id
            }
            if tuple(sorted(by_preparation)) != development_ids:
                raise ValueError("odd Jacobian development slopes are incomplete")
            target = np.asarray(
                tuple(by_preparation[value].slope for value in development_ids),
                dtype=np.float64,
            )
            alpha = _select_alpha(
                full_design,
                target,
                development_ids,
                config.ridge_candidates,
                family_scales,
                config,
            )
            specifications = (
                (
                    OddJacobianModelKind.STATE_INDEPENDENT,
                    np.ones((len(development_ids), 1), dtype=np.float64),
                    0.0,
                    ("intercept",),
                    (),
                    (),
                ),
                (
                    OddJacobianModelKind.STATE_DEPENDENT,
                    full_design,
                    alpha,
                    (
                        "intercept",
                        *(f"state-{value.coordinate_id}" for value in active_state_coordinates),
                    ),
                    state_centres,
                    state_scales,
                ),
            )
            for kind, design, model_alpha, terms, model_centres, model_scales in specifications:
                coefficients = _ridge(design, target, model_alpha)
                coefficient_records = tuple(
                    sorted(
                        (
                            OddJacobianCoefficient(
                                coefficient_id=(
                                    f"coefficient.{kind.value.lower().replace('_', '-')}."
                                    f"{action.port_id}.h{horizon_s}."
                                    f"{receiver_id}.{term_id}"
                                ),
                                receiver_coordinate_id=receiver_id,
                                term_id=term_id,
                                value=_decimal(coefficients[term_index, receiver_index]),
                                native_unit=f"{receiver_units[receiver_id]}/({action_units[action.port_id]})",
                            )
                            for term_index, term_id in enumerate(terms)
                            for receiver_index, receiver_id in enumerate(receiver_ids)
                        ),
                        key=lambda value: value.coefficient_id,
                    )
                )
                model_condition = float(np.linalg.cond(design))
                models.append(
                    OddJacobianModel(
                        model_id=(
                            f"model.{dataset.dataset_id}.{kind.value.lower().replace('_', '-')}."
                            f"{action.port_id}.h{horizon_s}"
                        ),
                        method_key=ODD_RELATIONAL_JACOBIAN_METHOD_KEY,
                        method_version=ODD_RELATIONAL_JACOBIAN_METHOD_VERSION,
                        model_kind=kind,
                        port_id=action.port_id,
                        action_unit=action.native_unit,
                        horizon_s=horizon_s,
                        state_centres=model_centres,
                        state_scales=model_scales,
                        receiver_coordinate_ids=receiver_ids,
                        family_scales=family_scale_records,
                        coefficients=coefficient_records,
                        ridge_alpha=_decimal(model_alpha),
                        design_rank=design.shape[1],
                        design_condition_number=_decimal(model_condition),
                        development_preparation_ids=development_ids,
                    )
                )
    return OddJacobianFit(
        fit_id=f"fit.{dataset.dataset_id}.odd-relational-jacobian",
        dataset=ObjectIdentity.from_record(dataset.dataset_id, dataset),
        config=ObjectIdentity.from_record(config.config_id, config),
        models=tuple(sorted(models, key=lambda value: value.model_id)),
        development_preparation_ids=development_ids,
        evaluation_preparation_ids=evaluation_ids,
    )


def _coefficient_matrix(model: OddJacobianModel) -> np.ndarray:
    terms = ("intercept",) + tuple(f"state-{value.value_id}" for value in model.state_centres)
    values = {
        (value.term_id, value.receiver_coordinate_id): float(value.value)
        for value in model.coefficients
    }
    return np.asarray(
        tuple(
            tuple(values[(term_id, receiver_id)] for receiver_id in model.receiver_coordinate_ids)
            for term_id in terms
        ),
        dtype=np.float64,
    )


def predict_odd_jacobian_slope(
    model: OddJacobianModel,
    state_values: tuple[NamedDecimal, ...],
) -> npt.NDArray[np.float64]:
    """Return the receiver-per-native-action slope at one causal state."""

    if model.model_kind is OddJacobianModelKind.STATE_INDEPENDENT:
        design = np.ones(1, dtype=np.float64)
    else:
        expected = tuple(value.value_id for value in model.state_centres)
        state = _selected_values(state_values, expected)
        centres = np.asarray(tuple(float(value.value) for value in model.state_centres))
        scales = np.asarray(tuple(float(value.value) for value in model.state_scales))
        observed_units = {
            value.value_id: value.unit for value in state_values if value.value_id in expected
        }
        expected_units = {value.value_id: value.unit for value in model.state_centres}
        if observed_units != expected_units:
            raise ValueError("odd Jacobian prediction state unit mismatch")
        design = np.concatenate(([1.0], (state - centres) / scales))
    return np.asarray(design @ _coefficient_matrix(model), dtype=np.float64)


def predict_odd_response(
    models: tuple[OddJacobianModel, ...],
    *,
    horizon_s: int,
    state_values: tuple[NamedDecimal, ...],
    action_values: tuple[NamedDecimal, ...],
    model_kind: OddJacobianModelKind,
) -> tuple[NamedDecimal, ...]:
    """Contract the local Jacobian with native action; zero is exact."""

    selected = tuple(
        value for value in models if value.horizon_s == horizon_s and value.model_kind is model_kind
    )
    if not selected:
        raise ValueError("odd Jacobian prediction has no model for the requested horizon")
    by_port = {value.port_id: value for value in selected}
    if len(by_port) != len(selected):
        raise ValueError("odd Jacobian prediction contains duplicate port models")
    provided = {value.value_id: value for value in action_values}
    if set(provided) != set(by_port):
        raise ValueError("odd Jacobian prediction action chart is incomplete")
    response = np.zeros(len(selected[0].receiver_coordinate_ids), dtype=np.float64)
    for port_id, model in by_port.items():
        action = provided[port_id]
        if action.unit != model.action_unit:
            raise ValueError("odd Jacobian prediction action unit mismatch")
        if action.value == 0:
            continue
        response += float(action.value) * predict_odd_jacobian_slope(model, state_values)
    receiver_units = {
        value.receiver_coordinate_id: value.native_unit.split("/(", maxsplit=1)[0]
        for value in selected[0].coefficients
        if value.term_id == "intercept"
    }
    return tuple(
        NamedDecimal(
            value_id=receiver_id,
            value=Decimal(0) if np.all(response == 0) else _decimal(response[index]),
            unit=receiver_units[receiver_id],
        )
        for index, receiver_id in enumerate(selected[0].receiver_coordinate_ids)
    )


def odd_jacobian_triplet_arrays(
    dataset: OddJacobianDataset,
    config: OddRelationalJacobianConfig,
) -> tuple[tuple[str, DataSplit, int, str, np.ndarray, np.ndarray], ...]:
    """Expose checked slopes/even components for adjudication and diagnostics."""

    return tuple(
        (
            value.preparation_id,
            value.split,
            value.horizon_s,
            value.port_id,
            value.slope.copy(),
            value.even.copy(),
        )
        for value in _triplets(dataset, config)
    )


__all__ = [
    "ODD_RELATIONAL_JACOBIAN_METHOD_KEY",
    "ODD_RELATIONAL_JACOBIAN_METHOD_VERSION",
    "OddJacobianActionCoordinate",
    "OddJacobianBranchObservation",
    "OddJacobianCoefficient",
    "OddJacobianCoordinate",
    "OddJacobianDataset",
    "OddJacobianFit",
    "OddJacobianModel",
    "OddJacobianModelKind",
    "OddJacobianReceiverCoordinate",
    "OddRelationalJacobianConfig",
    "fit_odd_relational_jacobian",
    "odd_jacobian_triplet_arrays",
    "predict_odd_jacobian_slope",
    "predict_odd_response",
]
