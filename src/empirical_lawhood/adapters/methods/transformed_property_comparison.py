"Typed, compatibility-applied property transport comparisons.\n\nThe operands encode unit, frame and receiver-direction transforms. The eight\npublic property kinds retain explicit scientific algorithms and fingerprinted\nnormalization operands. Compatibility-map identities alone cannot supply these\noperands or apply the transforms.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import ClassVar, Callable

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ScientificStatus

from .interval_property_comparison import PropertyComparisonKind, IntervalPropertyComparisonResult


TRANSFORMED_PROPERTY_COMPARISON_METHOD_KEY = 'method.transformed-property-comparison'
TRANSFORMED_PROPERTY_COMPARISON_METHOD_VERSION = "1.0.0"
TRANSFORMED_PROPERTY_COMPARISON_IMPLEMENTATION_SHA256 = sha256(
    b'empirical-lawhood/transformed-property-comparison:typed-maps-eight-distinct-algorithms'
).hexdigest()
_KINDS = tuple(sorted(PropertyComparisonKind, key=lambda value: value.value))


def _owned_id(prefix: str, kind: PropertyComparisonKind) -> str:
    label = kind.value.lower().replace("_", "-")
    return f"transformed-property-comparison.{prefix}.{label}"


TRANSFORMED_PROPERTY_COMPARISON_ALGORITHM_IDS = tuple(_owned_id("algorithm", value) for value in _KINDS)
TRANSFORMED_PROPERTY_COMPARISON_UNCERTAINTY_RULE_IDS = tuple(
    _owned_id("uncertainty", value) for value in _KINDS
)
TRANSFORMED_PROPERTY_COMPARISON_DECISIVE_FALSIFIER_IDS = tuple(
    _owned_id("falsifier", value) for value in _KINDS
)


@dataclass(frozen=True, slots=True)
class TransformedPropertyComparisonMethodSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/transformed-property-comparison-method-spec'

    spec_id: str
    method_key: str
    method_version: str
    implementation_sha256: str
    supported_comparison_kinds: tuple[PropertyComparisonKind, ...]
    algorithm_ids: tuple[str, ...]
    uncertainty_rule_ids: tuple[str, ...]
    decisive_falsifier_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.spec_id, field_name="spec_id")
        validate_stable_id(self.method_key, field_name="method_key")
        validate_semantic_version(self.method_version)
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if (
            self.method_key != TRANSFORMED_PROPERTY_COMPARISON_METHOD_KEY
            or self.method_version != TRANSFORMED_PROPERTY_COMPARISON_METHOD_VERSION
            or self.implementation_sha256 != TRANSFORMED_PROPERTY_COMPARISON_IMPLEMENTATION_SHA256
        ):
            raise ValueError('transformed property comparison spec names another implementation')
        if self.supported_comparison_kinds != _KINDS:
            raise ValueError('transformed property comparison requires the exact eight-kind roster')
        expected = (
            (self.algorithm_ids, TRANSFORMED_PROPERTY_COMPARISON_ALGORITHM_IDS, "algorithm_ids"),
            (
                self.uncertainty_rule_ids,
                TRANSFORMED_PROPERTY_COMPARISON_UNCERTAINTY_RULE_IDS,
                "uncertainty_rule_ids",
            ),
            (
                self.decisive_falsifier_ids,
                TRANSFORMED_PROPERTY_COMPARISON_DECISIVE_FALSIFIER_IDS,
                "decisive_falsifier_ids",
            ),
        )
        for observed, required, label in expected:
            require_sorted_unique_strings(observed, field_name=label, allow_empty=False)
            if observed != required:
                raise ValueError(f"transformed property comparison {label} differs")


@dataclass(frozen=True, slots=True)
class PropertyCompatibilityAxisMap(CanonicalRecord):
    """One authenticated affine target-to-source normalization contract.

    ``target_value = scale * source_value + offset``.  The evaluator inverts
    that relation before comparison.  A negative scale is required exactly
    when source and target receiver directions oppose.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/property-compatibility-axis-map'

    map_id: str
    value_id: str
    source_unit: str
    target_unit: str
    source_frame: str
    target_frame: str
    source_direction: int
    target_direction: int
    scale: Decimal
    offset: Decimal
    resolution_in_source_unit: Decimal

    def __post_init__(self) -> None:
        for name in ("map_id", "value_id", "source_frame", "target_frame"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("source_unit", "target_unit"):
            validate_nonempty(getattr(self, name), field_name=name)
        if self.source_direction not in {-1, 1} or self.target_direction not in {-1, 1}:
            raise ValueError("property axis directions must be signed")
        if self.scale == 0 or self.resolution_in_source_unit < 0:
            raise ValueError("property axis map has a singular scale or negative resolution")
        same_direction = self.source_direction == self.target_direction
        if (self.scale > 0) != same_direction:
            raise ValueError("property axis scale sign differs from receiver direction mapping")

    def normalize_target(self, value: Decimal) -> Decimal:
        return (value - self.offset) / self.scale

    def normalize_target_uncertainty(self, value: Decimal) -> Decimal:
        return value / abs(self.scale)


@dataclass(frozen=True, slots=True)
class PropertyOperandSeries(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/property-operand-series'

    series_id: str
    categorical_values: tuple[str, ...]
    metric_values: tuple[NamedDecimal, ...]
    uncertainty_values: tuple[NamedDecimal, ...]
    lineage_axis_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.series_id, field_name="series_id")
        require_sorted_unique_strings(
            self.categorical_values,
            field_name="categorical_values",
        )
        require_sorted_unique_ids(
            self.metric_values,
            attribute="value_id",
            field_name="metric_values",
        )
        require_sorted_unique_ids(
            self.uncertainty_values,
            attribute="value_id",
            field_name="uncertainty_values",
        )
        require_sorted_unique_strings(
            self.lineage_axis_ids,
            field_name="lineage_axis_ids",
        )
        if {value.value_id for value in self.metric_values} != {
            value.value_id for value in self.uncertainty_values
        }:
            raise ValueError("property series uncertainty roster differs from its metric roster")
        if any(value.value < 0 for value in self.uncertainty_values):
            raise ValueError("property series uncertainty cannot be negative")


@dataclass(frozen=True, slots=True)
class TransformedPropertyComparisonOperands(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/transformed-property-comparison-operands'

    operands_id: str
    comparison_kind: PropertyComparisonKind
    target_member_id: str
    property_id: str
    receiver_action_face_id: str
    predictive_level: str
    source_law_or_property: ObjectIdentity
    target_law_or_observation: ObjectIdentity
    domain_cell_ids: tuple[str, ...]
    excluded_field_ids: tuple[str, ...]
    compatibility_maps: tuple[PropertyCompatibilityAxisMap, ...]
    source: PropertyOperandSeries
    target: PropertyOperandSeries

    def __post_init__(self) -> None:
        for name in (
            "operands_id",
            "target_member_id",
            "property_id",
            "receiver_action_face_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.predictive_level not in {"CATEGORICAL", "METRIC", "DYNAMICAL"}:
            raise ValueError('transformed property comparison predictive level is unknown')
        require_sorted_unique_strings(
            self.domain_cell_ids,
            field_name="domain_cell_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.excluded_field_ids, field_name="excluded_field_ids")
        require_sorted_unique_ids(
            self.compatibility_maps,
            attribute="value_id",
            field_name="compatibility_maps",
        )
        source_ids = {value.value_id for value in self.source.metric_values}
        target_ids = {value.value_id for value in self.target.metric_values}
        map_ids = {value.value_id for value in self.compatibility_maps}
        if source_ids != target_ids or source_ids != map_ids:
            raise ValueError('transformed property comparison metric/map rosters differ')
        source_units = {value.value_id: value.unit for value in self.source.metric_values}
        target_units = {value.value_id: value.unit for value in self.target.metric_values}
        if any(
            source_units[value.value_id] != value.source_unit
            or target_units[value.value_id] != value.target_unit
            for value in self.compatibility_maps
        ):
            raise ValueError('transformed property comparison native units bypass their typed map')

    @property
    def face_key(self) -> tuple[str, str, str, str]:
        return (
            self.target_member_id,
            self.property_id,
            self.receiver_action_face_id,
            self.predictive_level,
        )


@dataclass(frozen=True, slots=True)
class TransformedPropertyComparisonResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/transformed-property-comparison-result'

    result_id: str
    method_spec: ObjectIdentity
    operands: ObjectIdentity
    algorithm_id: str
    uncertainty_rule_id: str
    comparison_kind: PropertyComparisonKind
    face_key: tuple[str, str, str, str]
    normalized_target_values: tuple[NamedDecimal, ...]
    metric_differences: tuple[NamedDecimal, ...]
    property_preserved: bool | None
    achieved_dependence_class: str
    disposition: ScientificStatus
    decisive_falsifier_id: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("result_id", "algorithm_id", "uncertainty_rule_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if len(self.face_key) != 4:
            raise ValueError('transformed property comparison result lacks its face key')
        require_sorted_unique_ids(
            self.normalized_target_values,
            attribute="value_id",
            field_name="normalized_target_values",
        )
        require_sorted_unique_ids(
            self.metric_differences,
            attribute="value_id",
            field_name="metric_differences",
        )
        validate_stable_id(self.achieved_dependence_class, field_name="achieved_dependence_class")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected = {
            ScientificStatus.SUPPORTED: (True, None),
            ScientificStatus.NOT_SUPPORTED: (False, self.decisive_falsifier_id),
            ScientificStatus.UNEVALUABLE: (None, None),
        }
        if self.disposition not in expected:
            raise ValueError('transformed property comparison disposition is unsupported')
        preserved, falsifier = expected[self.disposition]
        if self.property_preserved is not preserved:
            raise ValueError('transformed property comparison verdict/disposition differ')
        if self.disposition is ScientificStatus.NOT_SUPPORTED:
            if falsifier is None:
                raise ValueError('opposed transformed property comparison lacks its falsifier')
            validate_stable_id(falsifier, field_name="decisive_falsifier_id")
        elif self.decisive_falsifier_id is not None:
            raise ValueError('non-opposed transformed property comparison fabricates a falsifier')


@dataclass(frozen=True, slots=True)
class _NormalizedOperands:
    source: dict[str, Decimal]
    target: dict[str, Decimal]
    source_uncertainty: dict[str, Decimal]
    target_uncertainty: dict[str, Decimal]
    resolution: dict[str, Decimal]
    source_categories: tuple[str, ...]
    target_categories: tuple[str, ...]


def _normalize(operands: TransformedPropertyComparisonOperands) -> _NormalizedOperands:
    source = {value.value_id: value.value for value in operands.source.metric_values}
    target_native = {value.value_id: value.value for value in operands.target.metric_values}
    source_uncertainty = {
        value.value_id: value.value for value in operands.source.uncertainty_values
    }
    target_uncertainty_native = {
        value.value_id: value.value for value in operands.target.uncertainty_values
    }
    maps = {value.value_id: value for value in operands.compatibility_maps}
    return _NormalizedOperands(
        source=source,
        target={key: maps[key].normalize_target(value) for key, value in target_native.items()},
        source_uncertainty=source_uncertainty,
        target_uncertainty={
            key: maps[key].normalize_target_uncertainty(value)
            for key, value in target_uncertainty_native.items()
        },
        resolution={key: value.resolution_in_source_unit for key, value in maps.items()},
        source_categories=operands.source.categorical_values,
        target_categories=operands.target.categorical_values,
    )


def _classify(value: Decimal, uncertainty: Decimal, resolution: Decimal) -> int | None:
    lower = value - uncertainty
    upper = value + uncertainty
    if lower > resolution:
        return 1
    if upper < -resolution:
        return -1
    if lower >= -resolution and upper <= resolution:
        return 0
    return None


def _classes(values: _NormalizedOperands) -> tuple[tuple[int, int], ...] | None:
    result = tuple(
        (
            _classify(
                values.source[key],
                values.source_uncertainty[key],
                values.resolution[key],
            ),
            _classify(
                values.target[key],
                values.target_uncertainty[key],
                values.resolution[key],
            ),
        )
        for key in sorted(values.source)
    )
    if any(left is None or right is None for left, right in result):
        return None
    definite: list[tuple[int, int]] = []
    for left, right in result:
        assert left is not None and right is not None
        definite.append((left, right))
    return tuple(definite)


def _sign_algorithm(values: _NormalizedOperands) -> bool | None:
    classes = _classes(values)
    return None if not classes else all(left == right for left, right in classes)


def _order_algorithm(values: _NormalizedOperands) -> bool | None:
    keys = sorted(values.source)
    if len(keys) < 2:
        return None
    comparisons: list[tuple[int | None, int | None]] = []
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            resolution = values.resolution[left] + values.resolution[right]
            comparisons.append(
                (
                    _classify(
                        values.source[right] - values.source[left],
                        values.source_uncertainty[right] + values.source_uncertainty[left],
                        resolution,
                    ),
                    _classify(
                        values.target[right] - values.target[left],
                        values.target_uncertainty[right] + values.target_uncertainty[left],
                        resolution,
                    ),
                )
            )
    if any(left is None or right is None for left, right in comparisons):
        return None
    return all(left == right for left, right in comparisons)


def _rank_null_algorithm(values: _NormalizedOperands) -> bool | None:
    classes = _classes(values)
    if not classes:
        return None
    return tuple(left == 0 for left, _ in classes) == tuple(right == 0 for _, right in classes)


def _history_dependence_algorithm(values: _NormalizedOperands) -> bool | None:
    classes = _classes(values)
    if not values.source_categories or not values.target_categories or not classes:
        return None
    source_dependent = tuple(left != 0 for left, _ in classes)
    target_dependent = tuple(right != 0 for _, right in classes)
    return (
        values.source_categories == values.target_categories
        and source_dependent == target_dependent
    )


def _receiver_conflict_algorithm(values: _NormalizedOperands) -> bool | None:
    classes = _classes(values)
    if not values.source_categories or not values.target_categories or not classes:
        return None
    source_conflict = len(set(left for left, _ in classes if left != 0)) > 1
    target_conflict = len(set(right for _, right in classes if right != 0)) > 1
    return (
        values.source_categories == values.target_categories and source_conflict == target_conflict
    )


def _reachability_algorithm(values: _NormalizedOperands) -> bool | None:
    classes = _classes(values)
    if not values.source_categories or not values.target_categories or not classes:
        return None
    source_image = tuple(left != 0 for left, _ in classes)
    target_image = tuple(right != 0 for _, right in classes)
    return values.source_categories == values.target_categories and source_image == target_image


def _branch_fragility_algorithm(values: _NormalizedOperands) -> bool | None:
    if not values.source_categories or not values.target_categories:
        return None
    if values.source:
        ordered = _order_algorithm(values)
        if ordered is None:
            return None
        return values.source_categories == values.target_categories and ordered
    return values.source_categories == values.target_categories


def _safety_constraint_algorithm(values: _NormalizedOperands) -> bool | None:
    classes = _classes(values)
    if not values.source_categories or not values.target_categories or not classes:
        return None
    source_safe = tuple(left >= 0 for left, _ in classes)
    target_safe = tuple(right >= 0 for _, right in classes)
    return values.source_categories == values.target_categories and source_safe == target_safe


_ALGORITHMS: dict[
    PropertyComparisonKind,
    Callable[[_NormalizedOperands], bool | None],
] = {
    PropertyComparisonKind.SIGN: _sign_algorithm,
    PropertyComparisonKind.ORDER: _order_algorithm,
    PropertyComparisonKind.RANK_NULL: _rank_null_algorithm,
    PropertyComparisonKind.HISTORY_DEPENDENCE: _history_dependence_algorithm,
    PropertyComparisonKind.RECEIVER_CONFLICT: _receiver_conflict_algorithm,
    PropertyComparisonKind.REACHABILITY: _reachability_algorithm,
    PropertyComparisonKind.BRANCH_FRAGILITY: _branch_fragility_algorithm,
    PropertyComparisonKind.SAFETY_CONSTRAINT: _safety_constraint_algorithm,
}


def _dependence(operands: TransformedPropertyComparisonOperands) -> str:
    source = set(operands.source.lineage_axis_ids)
    target = set(operands.target.lineage_axis_ids)
    if not source or not target:
        return "dependence.unresolved-lineage"
    if source == target:
        return "dependence.shared-lineage"
    if source.isdisjoint(target):
        return "dependence.disjoint-declared-lineage"
    return "dependence.partially-shared-lineage"


def evaluate_property_comparison(
    *,
    spec: TransformedPropertyComparisonMethodSpec,
    operands: TransformedPropertyComparisonOperands,
) -> TransformedPropertyComparisonResult:
    """Normalize through the authenticated map and dispatch one kind-local test."""

    index = spec.supported_comparison_kinds.index(operands.comparison_kind)
    normalized = _normalize(operands)
    preserved = _ALGORITHMS[operands.comparison_kind](normalized)
    if preserved is True:
        disposition = ScientificStatus.SUPPORTED
        falsifier = None
        reasons = ('TRANSFORMED_PROPERTY_COMPARISON_SUPPORTED',)
    elif preserved is False:
        disposition = ScientificStatus.NOT_SUPPORTED
        falsifier = spec.decisive_falsifier_ids[index]
        reasons = ('TRANSFORMED_PROPERTY_COMPARISON_DECISIVE_FALSIFIER',)
    else:
        disposition = ScientificStatus.UNEVALUABLE
        falsifier = None
        reasons = ('TRANSFORMED_PROPERTY_COMPARISON_OPERANDS_UNEVALUABLE',)
    maps = {value.value_id: value for value in operands.compatibility_maps}
    normalized_values = tuple(
        NamedDecimal(value_id=key, value=value, unit=maps[key].source_unit)
        for key, value in sorted(normalized.target.items())
    )
    differences = tuple(
        NamedDecimal(
            value_id=f"difference.{key}",
            value=normalized.target[key] - normalized.source[key],
            unit=maps[key].source_unit,
        )
        for key in sorted(normalized.source)
    )
    spec_identity = ObjectIdentity.from_record(spec.spec_id, spec)
    operands_identity = ObjectIdentity.from_record(operands.operands_id, operands)
    result_seed = {
        "spec": spec_identity,
        "operands": operands_identity,
        "algorithm": spec.algorithm_ids[index],
        "preserved": preserved,
    }
    return TransformedPropertyComparisonResult(
        result_id=(
            'transformed-property-comparison-result.'
            f"{sha256(canonical_json_bytes(result_seed)).hexdigest()[:32]}"
        ),
        method_spec=spec_identity,
        operands=operands_identity,
        algorithm_id=spec.algorithm_ids[index],
        uncertainty_rule_id=spec.uncertainty_rule_ids[index],
        comparison_kind=operands.comparison_kind,
        face_key=operands.face_key,
        normalized_target_values=normalized_values,
        metric_differences=differences,
        property_preserved=preserved,
        achieved_dependence_class=_dependence(operands),
        disposition=disposition,
        decisive_falsifier_id=falsifier,
        reason_codes=reasons,
    )


def bridge_transformed_property_result_to_interval(
    *,
    result: TransformedPropertyComparisonResult,
    operands: TransformedPropertyComparisonOperands,
) -> IntervalPropertyComparisonResult:
    'Project an adjudicated transformed result into the interval result carrier.\n\n    The projection preserves the verdict and accepts only the exact operand\n    identity adjudicated by the transformed property comparison method.\n    '

    if result.operands != ObjectIdentity.from_record(operands.operands_id, operands):
        raise ValueError('transformed property comparison bridge substitutes its exact operands')
    return IntervalPropertyComparisonResult(
        result_id=f"{result.result_id}.interval-bridge",
        method_spec=result.method_spec,
        operands=result.operands,
        comparison_kind=result.comparison_kind,
        face_key=result.face_key,
        source_summary_values=operands.source.categorical_values,
        target_summary_values=operands.target.categorical_values,
        metric_differences=result.metric_differences,
        property_preserved=result.property_preserved,
        achieved_dependence_class=result.achieved_dependence_class,
        disposition=result.disposition,
        decisive_falsifier_id=result.decisive_falsifier_id,
        reason_codes=tuple(sorted((*result.reason_codes, 'TRANSFORMED_TO_INTERVAL_RESULT_PROJECTION'))),
    )


__all__ = [
    'TRANSFORMED_PROPERTY_COMPARISON_ALGORITHM_IDS',
    'TRANSFORMED_PROPERTY_COMPARISON_DECISIVE_FALSIFIER_IDS',
    'TRANSFORMED_PROPERTY_COMPARISON_IMPLEMENTATION_SHA256',
    'TRANSFORMED_PROPERTY_COMPARISON_METHOD_KEY',
    'TRANSFORMED_PROPERTY_COMPARISON_METHOD_VERSION',
    'TRANSFORMED_PROPERTY_COMPARISON_UNCERTAINTY_RULE_IDS',
    'TransformedPropertyComparisonMethodSpec',
    'TransformedPropertyComparisonOperands',
    'TransformedPropertyComparisonResult',
    'PropertyCompatibilityAxisMap',
    'PropertyOperandSeries',
    'bridge_transformed_property_result_to_interval',
    'evaluate_property_comparison',
]
