"""Finite coordinate-collision and reduced-comparator tournament for physical scale morphism."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .contracts import PhysicalScaleMorphismComparatorKind, PHYSICAL_SCALE_MORPHISM_COORDINATE_IDS


class PhysicalScaleMorphismCoordinateDisposition(StrEnum):
    COORDINATE_NECESSITY_IDENTIFIED = "COORDINATE_NECESSITY_IDENTIFIED"
    COORDINATE_PREDICTION_OPPOSED = "COORDINATE_PREDICTION_OPPOSED"
    TASK_SATURATED_BY_ACTION_ONLY_OR_WILDCARD = "TASK_SATURATED_BY_ACTION_ONLY_OR_WILDCARD"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismCollisionMember(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-collision-member'

    member_id: str
    complete_unit_id: str
    denominator_id: str
    history_id: str
    action_id: str
    receiver_id: str
    horizon_id: str
    categorical_label: str | None

    def __post_init__(self) -> None:
        for name in (
            "member_id",
            "complete_unit_id",
            "denominator_id",
            "history_id",
            "action_id",
            "receiver_id",
            "horizon_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.categorical_label is not None:
            validate_stable_id(self.categorical_label, field_name="categorical_label")

    def coordinate(self, coordinate_id: str) -> str:
        return {
            "D": self.denominator_id,
            "H": self.history_id,
            "A": self.action_id,
            "R": self.receiver_id,
            "tau": self.horizon_id,
        }[coordinate_id]


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismCoordinateCollision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-coordinate-collision'

    collision_id: str
    challenged_coordinate_id: str
    left: PhysicalScaleMorphismCollisionMember
    right: PhysicalScaleMorphismCollisionMember
    outcome_required_to_differ: bool
    evaluation_held_out: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.collision_id, field_name="collision_id")
        if self.challenged_coordinate_id not in PHYSICAL_SCALE_MORPHISM_COORDINATE_IDS:
            raise ValueError("collision challenges an unknown law coordinate")
        equal_coordinates = tuple(
            coordinate_id
            for coordinate_id in PHYSICAL_SCALE_MORPHISM_COORDINATE_IDS
            if self.left.coordinate(coordinate_id) == self.right.coordinate(coordinate_id)
        )
        if self.challenged_coordinate_id in equal_coordinates:
            raise ValueError("collision does not vary its challenged coordinate")
        if any(
            coordinate_id not in equal_coordinates
            for coordinate_id in PHYSICAL_SCALE_MORPHISM_COORDINATE_IDS
            if coordinate_id != self.challenged_coordinate_id
        ):
            raise ValueError("collision varies more than its challenged coordinate")
        if not self.evaluation_held_out:
            raise ValueError("coordinate-necessity collision must be held out")

    @property
    def evaluable(self) -> bool:
        return self.left.categorical_label is not None and self.right.categorical_label is not None

    @property
    def observed_difference(self) -> bool | None:
        if not self.evaluable:
            return None
        return self.left.categorical_label != self.right.categorical_label


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismComparatorPrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-comparator-prediction'

    prediction_id: str
    member_id: str
    categorical_label: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.prediction_id, field_name="prediction_id")
        validate_stable_id(self.member_id, field_name="member_id")
        if self.categorical_label is not None:
            validate_stable_id(self.categorical_label, field_name="categorical_label")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismComparatorEncoding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-comparator-encoding'

    encoding_id: str
    kind: PhysicalScaleMorphismComparatorKind
    dependency_coordinate_ids: tuple[str, ...]
    predictions: tuple[PhysicalScaleMorphismComparatorPrediction, ...]
    fitted_complete_unit_ids: tuple[str, ...]
    evaluation_outcome_count_at_freeze: int

    def __post_init__(self) -> None:
        validate_stable_id(self.encoding_id, field_name="encoding_id")
        expected = tuple(
            value for value in PHYSICAL_SCALE_MORPHISM_COORDINATE_IDS if value in self.dependency_coordinate_ids
        )
        if self.dependency_coordinate_ids != expected:
            raise ValueError("comparator dependencies are unknown or noncanonical")
        require_sorted_unique_ids(
            self.predictions, attribute="prediction_id", field_name="predictions"
        )
        if len({value.member_id for value in self.predictions}) != len(self.predictions):
            raise ValueError("comparator contains duplicate member predictions")
        require_sorted_unique_strings(
            self.fitted_complete_unit_ids,
            field_name="fitted_complete_unit_ids",
            allow_empty=self.kind in {PhysicalScaleMorphismComparatorKind.WILDCARD},
        )
        if self.evaluation_outcome_count_at_freeze:
            raise ValueError("comparator encoding cannot inspect evaluation outcomes")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismComparatorScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-comparator-score'

    score_id: str
    encoding_id: str
    kind: PhysicalScaleMorphismComparatorKind
    evaluable_member_count: int
    mismatch_member_ids: tuple[str, ...]
    abstention_member_ids: tuple[str, ...]
    exact: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.score_id, field_name="score_id")
        validate_stable_id(self.encoding_id, field_name="encoding_id")
        for name in ("mismatch_member_ids", "abstention_member_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if self.evaluable_member_count < 0:
            raise ValueError("evaluable comparator member count must be nonnegative")
        expected = (
            self.evaluable_member_count > 0
            and not self.mismatch_member_ids
            and not self.abstention_member_ids
        )
        if self.exact != expected:
            raise ValueError("comparator exactness is not data-derived")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismCoordinateComparatorPanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-coordinate-comparator-panel'

    panel_id: str
    selected_encoding_id: str
    selected_coordinate_ids: tuple[str, ...]
    active_coordinate_ids: tuple[str, ...]
    inactive_coordinate_ids: tuple[str, ...]
    collision_ids: tuple[str, ...]
    unevaluable_collision_ids: tuple[str, ...]
    prediction_opposed_collision_ids: tuple[str, ...]
    score_by_kind: tuple[PhysicalScaleMorphismComparatorScore, ...]
    disposition: PhysicalScaleMorphismCoordinateDisposition

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        validate_stable_id(self.selected_encoding_id, field_name="selected_encoding_id")
        for name in (
            "selected_coordinate_ids",
            "active_coordinate_ids",
            "inactive_coordinate_ids",
        ):
            values = getattr(self, name)
            expected = tuple(value for value in PHYSICAL_SCALE_MORPHISM_COORDINATE_IDS if value in values)
            if values != expected:
                raise ValueError(f"{name} is unknown or noncanonical")
        if set(self.active_coordinate_ids) & set(self.inactive_coordinate_ids):
            raise ValueError("active and inactive coordinate sets overlap")
        require_sorted_unique_strings(
            self.collision_ids, field_name="collision_ids", allow_empty=False
        )
        require_sorted_unique_strings(
            self.unevaluable_collision_ids, field_name="unevaluable_collision_ids"
        )
        require_sorted_unique_strings(
            self.prediction_opposed_collision_ids,
            field_name="prediction_opposed_collision_ids",
        )
        if not (
            set(self.unevaluable_collision_ids) | set(self.prediction_opposed_collision_ids)
        ).issubset(self.collision_ids):
            raise ValueError("collision disposition is outside the frozen roster")
        if set(self.unevaluable_collision_ids) & set(self.prediction_opposed_collision_ids):
            raise ValueError("one collision cannot be unevaluable and opposed")
        require_sorted_unique_ids(
            self.score_by_kind, attribute="score_id", field_name="score_by_kind"
        )


def evaluate_coordinate_comparators(
    *,
    panel_id: str,
    selected_encoding_id: str,
    collisions: tuple[PhysicalScaleMorphismCoordinateCollision, ...],
    encodings: tuple[PhysicalScaleMorphismComparatorEncoding, ...],
) -> PhysicalScaleMorphismCoordinateComparatorPanel:
    """Score frozen encodings and identify active/inactive law coordinates.

    The function never fits from evaluation outcomes.  It only compares the
    already frozen predictions with held-out collision labels.
    """

    require_sorted_unique_ids(collisions, attribute="collision_id", field_name="collisions")
    require_sorted_unique_ids(encodings, attribute="encoding_id", field_name="encodings")
    if len({value.kind for value in encodings}) != len(encodings):
        raise ValueError("coordinate tournament repeats a comparator kind")
    if {value.kind for value in encodings} != set(PhysicalScaleMorphismComparatorKind):
        raise ValueError("coordinate tournament lacks the frozen comparator lattice")
    by_encoding_id = {value.encoding_id: value for value in encodings}
    try:
        selected = by_encoding_id[selected_encoding_id]
    except KeyError as error:
        raise ValueError("selected comparator encoding is absent") from error
    members = {
        value.member_id: value
        for collision in collisions
        for value in (collision.left, collision.right)
    }
    evaluable_ids = {
        member_id for member_id, value in members.items() if value.categorical_label is not None
    }
    scores: list[PhysicalScaleMorphismComparatorScore] = []
    for encoding in encodings:
        predictions = {value.member_id: value.categorical_label for value in encoding.predictions}
        if set(predictions) != set(members):
            raise ValueError("comparator prediction roster differs from collision members")
        mismatch_ids = tuple(
            sorted(
                member_id
                for member_id in evaluable_ids
                if predictions[member_id] is not None
                and predictions[member_id] != members[member_id].categorical_label
            )
        )
        abstention_ids = tuple(
            sorted(member_id for member_id in evaluable_ids if predictions[member_id] is None)
        )
        scores.append(
            PhysicalScaleMorphismComparatorScore(
                score_id=f"{panel_id}.score.{encoding.kind.value.lower().replace('_', '-')}",
                encoding_id=encoding.encoding_id,
                kind=encoding.kind,
                evaluable_member_count=len(evaluable_ids),
                mismatch_member_ids=mismatch_ids,
                abstention_member_ids=abstention_ids,
                exact=bool(evaluable_ids) and not mismatch_ids and not abstention_ids,
            )
        )
    scores_tuple = tuple(sorted(scores, key=lambda value: value.score_id))
    score_by_encoding = {value.encoding_id: value for value in scores_tuple}
    observed_by_coordinate: dict[str, list[bool]] = {value: [] for value in PHYSICAL_SCALE_MORPHISM_COORDINATE_IDS}
    unevaluable = []
    prediction_opposed = []
    for collision in collisions:
        difference = collision.observed_difference
        if difference is None:
            unevaluable.append(collision.collision_id)
            continue
        observed_by_coordinate[collision.challenged_coordinate_id].append(difference)
        if difference != collision.outcome_required_to_differ:
            prediction_opposed.append(collision.collision_id)
    active = tuple(
        coordinate_id
        for coordinate_id in PHYSICAL_SCALE_MORPHISM_COORDINATE_IDS
        if observed_by_coordinate[coordinate_id] and any(observed_by_coordinate[coordinate_id])
    )
    inactive = tuple(
        coordinate_id
        for coordinate_id in PHYSICAL_SCALE_MORPHISM_COORDINATE_IDS
        if observed_by_coordinate[coordinate_id] and not any(observed_by_coordinate[coordinate_id])
    )
    saturation_kinds = {
        PhysicalScaleMorphismComparatorKind.ACTION_ONLY,
        PhysicalScaleMorphismComparatorKind.SATURATED_DEVELOPMENT_LOOKUP,
        PhysicalScaleMorphismComparatorKind.WILDCARD,
    }
    saturated = any(value.exact for value in scores_tuple if value.kind in saturation_kinds)
    strict_smaller = tuple(
        encoding
        for encoding in encodings
        if set(encoding.dependency_coordinate_ids) < set(selected.dependency_coordinate_ids)
    )
    necessity = (
        not unevaluable
        and score_by_encoding[selected_encoding_id].exact
        and selected.dependency_coordinate_ids == active
        and all(not score_by_encoding[value.encoding_id].exact for value in strict_smaller)
        and not saturated
    )
    if unevaluable:
        disposition = PhysicalScaleMorphismCoordinateDisposition.UNEVALUABLE
    elif saturated:
        disposition = PhysicalScaleMorphismCoordinateDisposition.TASK_SATURATED_BY_ACTION_ONLY_OR_WILDCARD
    elif prediction_opposed:
        disposition = PhysicalScaleMorphismCoordinateDisposition.COORDINATE_PREDICTION_OPPOSED
    elif necessity:
        disposition = PhysicalScaleMorphismCoordinateDisposition.COORDINATE_NECESSITY_IDENTIFIED
    else:
        disposition = PhysicalScaleMorphismCoordinateDisposition.UNEVALUABLE
    return PhysicalScaleMorphismCoordinateComparatorPanel(
        panel_id=panel_id,
        selected_encoding_id=selected_encoding_id,
        selected_coordinate_ids=selected.dependency_coordinate_ids,
        active_coordinate_ids=active,
        inactive_coordinate_ids=inactive,
        collision_ids=tuple(value.collision_id for value in collisions),
        unevaluable_collision_ids=tuple(sorted(unevaluable)),
        prediction_opposed_collision_ids=tuple(sorted(prediction_opposed)),
        score_by_kind=scores_tuple,
        disposition=disposition,
    )


__all__ = [
    'PhysicalScaleMorphismCollisionMember',
    'PhysicalScaleMorphismComparatorEncoding',
    'PhysicalScaleMorphismComparatorPrediction',
    'PhysicalScaleMorphismComparatorScore',
    'PhysicalScaleMorphismCoordinateCollision',
    'PhysicalScaleMorphismCoordinateComparatorPanel',
    'PhysicalScaleMorphismCoordinateDisposition',
    "evaluate_coordinate_comparators",
]
