"""Native-unit farthest-point ordering and worst-member utility evidence."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from enum import StrEnum
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .contracts import CanonicalVector, decimal_from_float


class FarthestPointSeedRule(StrEnum):
    LOWEST_POINT_ID = "LOWEST_POINT_ID"


class FarthestPointTieBreakRule(StrEnum):
    LOWEST_POINT_ID = "LOWEST_POINT_ID"


@dataclass(frozen=True, slots=True)
class FarthestPointConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/farthest-point-config'

    config_id: str
    coordinate_ids: tuple[str, ...]
    native_units: tuple[str, ...]
    coordinate_scales: tuple[Decimal, ...]
    seed_rule: FarthestPointSeedRule
    tie_break_rule: FarthestPointTieBreakRule

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_strings(
            self.coordinate_ids,
            field_name="coordinate_ids",
            allow_empty=False,
        )
        if len(self.native_units) != len(self.coordinate_ids) or any(
            not value.strip() for value in self.native_units
        ):
            raise ValueError("farthest-point config requires one native unit per coordinate")
        if len(self.coordinate_scales) != len(self.coordinate_ids):
            raise ValueError("farthest-point config requires one scale per coordinate")
        for value in self.coordinate_scales:
            validate_decimal(value, field_name="coordinate_scales", minimum=Decimal(0))
            if value <= 0:
                raise ValueError("farthest-point coordinate scales must be positive")


class MaximinAggregationRule(StrEnum):
    LITERAL_MINIMUM = "LITERAL_MINIMUM"


class MaximinTieBreakRule(StrEnum):
    UTILITY_DESCENDING_CANDIDATE_ID_ASCENDING = "UTILITY_DESCENDING_CANDIDATE_ID_ASCENDING"


@dataclass(frozen=True, slots=True)
class MaximinConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/maximin-config'

    config_id: str
    expected_member_ids: tuple[str, ...]
    utility_unit: str
    aggregation_rule: MaximinAggregationRule
    tie_break_rule: MaximinTieBreakRule

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_strings(
            self.expected_member_ids,
            field_name="expected_member_ids",
            allow_empty=False,
        )
        if not self.utility_unit.strip():
            raise ValueError("maximin utility unit must not be empty")


@dataclass(frozen=True, slots=True)
class NativeActionPoint(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/native-action-point'

    point_id: str
    action_word: ObjectIdentity
    coordinates: CanonicalVector

    def __post_init__(self) -> None:
        validate_stable_id(self.point_id, field_name="point_id")
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("native action point requires an exact ActionWord")


@dataclass(frozen=True, slots=True)
class FarthestPointOrder(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/farthest-point-order'

    order_id: str
    config: ObjectIdentity
    points: tuple[ObjectIdentity, ...]
    ordered_point_ids: tuple[str, ...]
    separation_distances: tuple[NamedDecimal, ...]
    coordinate_ids: tuple[str, ...]
    native_units: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.order_id, field_name="order_id")
        if self.config.object_schema != FarthestPointConfig.SCHEMA:
            raise ValueError("farthest-point order requires an exact config")
        require_sorted_unique_ids(self.points, attribute="object_id", field_name="points")
        require_sorted_unique_strings(
            tuple(sorted(self.ordered_point_ids)),
            field_name="ordered_point_ids",
        )
        require_sorted_unique_ids(
            self.separation_distances,
            attribute="value_id",
            field_name="separation_distances",
        )
        require_sorted_unique_strings(
            self.coordinate_ids,
            field_name="coordinate_ids",
            allow_empty=False,
        )
        if len(self.native_units) != len(self.coordinate_ids):
            raise ValueError("farthest-point order requires one native unit per coordinate")


@dataclass(frozen=True, slots=True)
class FarthestPointContinuationConfig(CanonicalRecord):
    """Continue maximin selection from an exact already-queried roster."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/farthest-point-continuation-config'

    continuation_config_id: str
    base_config: FarthestPointConfig
    already_queried_point_ids: tuple[str, ...]
    maximum_new_points: int

    def __post_init__(self) -> None:
        validate_stable_id(
            self.continuation_config_id,
            field_name="continuation_config_id",
        )
        require_sorted_unique_strings(
            self.already_queried_point_ids,
            field_name="already_queried_point_ids",
            allow_empty=False,
        )
        if self.maximum_new_points < 1:
            raise ValueError("farthest-point continuation must request new points")


@dataclass(frozen=True, slots=True)
class FarthestPointContinuationResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/farthest-point-continuation-result'

    result_id: str
    config: ObjectIdentity
    points: tuple[ObjectIdentity, ...]
    already_queried_point_ids: tuple[str, ...]
    selected_point_ids: tuple[str, ...]
    separation_distances: tuple[NamedDecimal, ...]
    exhausted: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.config.object_schema != FarthestPointContinuationConfig.SCHEMA:
            raise ValueError("farthest-point continuation result binds another config")
        require_sorted_unique_ids(self.points, attribute="object_id", field_name="points")
        require_sorted_unique_strings(
            self.already_queried_point_ids,
            field_name="already_queried_point_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            tuple(sorted(self.selected_point_ids)),
            field_name="selected_point_ids",
        )
        require_sorted_unique_ids(
            self.separation_distances,
            attribute="value_id",
            field_name="separation_distances",
        )
        if len(self.separation_distances) != len(self.selected_point_ids):
            raise ValueError("continuation distances differ from selected points")
        if set(self.already_queried_point_ids) & set(self.selected_point_ids):
            raise ValueError("continuation reselects an already queried point")


@dataclass(frozen=True, slots=True)
class WorstMemberUtility(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/worst-member-utility'

    utility_id: str
    candidate_id: str
    member_utilities: tuple[NamedDecimal, ...]
    worst_member_utility: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.utility_id, field_name="utility_id")
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        require_sorted_unique_ids(
            self.member_utilities,
            attribute="value_id",
            field_name="member_utilities",
        )
        if not self.member_utilities:
            raise ValueError("maximin candidate requires every member utility")
        if self.worst_member_utility.value != min(value.value for value in self.member_utilities):
            raise ValueError("worst-member utility is not the literal minimum")


@dataclass(frozen=True, slots=True)
class MaximinCandidateSet(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/maximin-candidate-set'

    candidate_set_id: str
    config: ObjectIdentity
    utilities: tuple[WorstMemberUtility, ...]
    ranked_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_set_id, field_name="candidate_set_id")
        if self.config.object_schema != MaximinConfig.SCHEMA:
            raise ValueError("maximin candidate set requires an exact config")
        require_sorted_unique_ids(self.utilities, attribute="utility_id", field_name="utilities")
        require_sorted_unique_strings(
            tuple(sorted(self.ranked_candidate_ids)),
            field_name="ranked_candidate_ids",
        )
        if set(self.ranked_candidate_ids) != {value.candidate_id for value in self.utilities}:
            raise ValueError("maximin ranking differs from candidate utility roster")


@dataclass(frozen=True, slots=True)
class FarthestPointMaximinService:
    def farthest_order(
        self,
        *,
        order_id: str,
        points: tuple[NativeActionPoint, ...],
        config: FarthestPointConfig,
    ) -> FarthestPointOrder:
        ordered_points = tuple(sorted(points, key=lambda value: value.point_id))
        if not ordered_points:
            raise ValueError("farthest-point ordering requires candidates")
        coordinate_ids = config.coordinate_ids
        if any(value.coordinates.coordinate_ids != coordinate_ids for value in ordered_points):
            raise ValueError("farthest-point candidates use different coordinate bases")
        scales = np.asarray(tuple(float(value) for value in config.coordinate_scales))
        remaining = {value.point_id: value for value in ordered_points}
        first_id = min(remaining)
        selected = [remaining.pop(first_id)]
        distances = [0.0]
        while remaining:
            scored = []
            for point_id, point in remaining.items():
                coordinates = point.coordinates.as_array() / scales
                minimum = min(
                    float(np.linalg.norm(coordinates - value.coordinates.as_array() / scales))
                    for value in selected
                )
                scored.append((minimum, point_id, point))
            maximum = max(value[0] for value in scored)
            _, point_id, point = min(
                (value for value in scored if np.isclose(value[0], maximum)),
                key=lambda value: value[1],
            )
            selected.append(point)
            distances.append(maximum)
            del remaining[point_id]
        return FarthestPointOrder(
            order_id=order_id,
            config=ObjectIdentity.from_record(config.config_id, config),
            points=tuple(
                ObjectIdentity.from_record(value.point_id, value) for value in ordered_points
            ),
            ordered_point_ids=tuple(value.point_id for value in selected),
            separation_distances=tuple(
                sorted(
                    (
                        NamedDecimal(
                            value_id=f"farthest-separation.{index:04d}",
                            value=decimal_from_float(distance),
                            unit="native-scaled-distance",
                        )
                        for index, distance in enumerate(distances)
                    ),
                    key=lambda value: value.value_id,
                )
            ),
            coordinate_ids=coordinate_ids,
            native_units=config.native_units,
        )

    def continue_farthest(
        self,
        *,
        result_id: str,
        points: tuple[NativeActionPoint, ...],
        config: FarthestPointContinuationConfig,
    ) -> FarthestPointContinuationResult:
        """Continue from the queried set without replaying the start-order rule."""

        ordered_points = tuple(sorted(points, key=lambda value: value.point_id))
        if not ordered_points:
            raise ValueError("farthest-point continuation requires candidates")
        by_id = {value.point_id: value for value in ordered_points}
        if len(by_id) != len(ordered_points):
            raise ValueError("farthest-point continuation candidate IDs are duplicated")
        if not set(config.already_queried_point_ids) <= set(by_id):
            raise ValueError("already queried point is absent from continuation candidates")
        base = config.base_config
        if any(value.coordinates.coordinate_ids != base.coordinate_ids for value in ordered_points):
            raise ValueError("continuation candidates use different coordinate bases")
        scales = base.coordinate_scales
        selected = [by_id[value] for value in config.already_queried_point_ids]
        remaining = {
            point_id: point
            for point_id, point in by_id.items()
            if point_id not in config.already_queried_point_ids
        }
        continuation: list[NativeActionPoint] = []
        distances: list[Decimal] = []
        while remaining and len(continuation) < config.maximum_new_points:
            scored: list[tuple[Decimal, str, NativeActionPoint]] = []
            for point_id, point in remaining.items():
                minimum = min(
                    sum(
                        (
                            ((left - right) / scale) * ((left - right) / scale)
                            for left, right, scale in zip(
                                point.coordinates.values,
                                prior.coordinates.values,
                                scales,
                                strict=True,
                            )
                        ),
                        Decimal(0),
                    )
                    for prior in selected
                )
                scored.append((minimum, point_id, point))
            maximum = max(value[0] for value in scored)
            _, point_id, point = min(
                (value for value in scored if value[0] == maximum),
                key=lambda value: value[1],
            )
            continuation.append(point)
            selected.append(point)
            with localcontext(Context(prec=50, rounding=ROUND_HALF_EVEN)) as context:
                distances.append(context.sqrt(maximum))
            del remaining[point_id]
        return FarthestPointContinuationResult(
            result_id=result_id,
            config=ObjectIdentity.from_record(config.continuation_config_id, config),
            points=tuple(
                ObjectIdentity.from_record(value.point_id, value) for value in ordered_points
            ),
            already_queried_point_ids=config.already_queried_point_ids,
            selected_point_ids=tuple(value.point_id for value in continuation),
            separation_distances=tuple(
                NamedDecimal(
                    value_id=f"continuation-separation.{index:04d}",
                    value=distance,
                    unit="native-scaled-distance",
                )
                for index, distance in enumerate(distances)
            ),
            exhausted=not remaining,
        )

    def maximin(
        self,
        *,
        candidate_set_id: str,
        candidate_member_utilities: tuple[tuple[str, tuple[NamedDecimal, ...]], ...],
        config: MaximinConfig,
    ) -> MaximinCandidateSet:
        utilities_list: list[WorstMemberUtility] = []
        for candidate_id, values in sorted(
            candidate_member_utilities,
            key=lambda value: value[0],
        ):
            ordered = tuple(sorted(values, key=lambda value: value.value_id))
            if tuple(value.value_id for value in ordered) != config.expected_member_ids:
                raise ValueError("maximin member utility roster is incomplete")
            if any(value.unit != config.utility_unit for value in ordered):
                raise ValueError("maximin member utilities change native unit")
            utilities_list.append(
                WorstMemberUtility(
                    utility_id=f"worst-member-utility.{candidate_id}",
                    candidate_id=candidate_id,
                    member_utilities=ordered,
                    worst_member_utility=NamedDecimal(
                        value_id=f"worst-member.{candidate_id}",
                        value=min(value.value for value in ordered),
                        unit=config.utility_unit,
                    ),
                )
            )
        utilities = tuple(utilities_list)
        ranked = tuple(
            value.candidate_id
            for value in sorted(
                utilities,
                key=lambda value: (-value.worst_member_utility.value, value.candidate_id),
            )
        )
        return MaximinCandidateSet(
            candidate_set_id=candidate_set_id,
            config=ObjectIdentity.from_record(config.config_id, config),
            utilities=utilities,
            ranked_candidate_ids=ranked,
        )
