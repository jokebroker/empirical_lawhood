"""Registered, method-owned property comparisons for structural transport."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar, cast

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ScientificStatus


class PropertyComparisonKind(StrEnum):
    SIGN = "SIGN"
    ORDER = "ORDER"
    RANK_NULL = "RANK_NULL"
    HISTORY_DEPENDENCE = "HISTORY_DEPENDENCE"
    RECEIVER_CONFLICT = "RECEIVER_CONFLICT"
    REACHABILITY = "REACHABILITY"
    BRANCH_FRAGILITY = "BRANCH_FRAGILITY"
    SAFETY_CONSTRAINT = "SAFETY_CONSTRAINT"


_KINDS = tuple(sorted(PropertyComparisonKind, key=lambda value: value.value))
INTERVAL_PROPERTY_COMPARISON_METHOD_KEY = 'method.interval-property-comparison'
INTERVAL_PROPERTY_COMPARISON_METHOD_VERSION = "1.0.0"
INTERVAL_PROPERTY_COMPARISON_IMPLEMENTATION_SHA256 = sha256(
    b'empirical-lawhood/interval-property-comparison:eight-kind-interval-aware'
).hexdigest()
INTERVAL_PROPERTY_COMPARISON_ALGORITHM_IDS = tuple(
    f"interval-property-comparison.algorithm.{value.value.lower().replace('_', '-')}" for value in _KINDS
)
INTERVAL_PROPERTY_COMPARISON_UNCERTAINTY_RULE_IDS = tuple(
    f"interval-property-comparison.uncertainty.{value.value.lower().replace('_', '-')}"
    for value in _KINDS
)
INTERVAL_PROPERTY_COMPARISON_TOLERANCE_IDS = tuple(
    f"interval-property-comparison.tolerance.{value.value.lower().replace('_', '-')}" for value in _KINDS
)
INTERVAL_PROPERTY_COMPARISON_DECISIVE_FALSIFIER_IDS = tuple(
    f"interval-property-comparison.falsifier.{value.value.lower().replace('_', '-')}" for value in _KINDS
)
_METHOD_OWNED_TOLERANCE_BY_KIND = {value: Decimal("0") for value in PropertyComparisonKind}


@dataclass(frozen=True, slots=True)
class IntervalPropertyComparisonMethodSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/interval-property-comparison-method-spec'

    spec_id: str
    method_key: str
    method_version: str
    implementation_sha256: str
    supported_comparison_kinds: tuple[PropertyComparisonKind, ...]
    algorithm_ids: tuple[str, ...]
    uncertainty_rule_ids: tuple[str, ...]
    tolerance_ids: tuple[str, ...]
    decisive_falsifier_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.spec_id, field_name="spec_id")
        validate_stable_id(self.method_key, field_name="method_key")
        validate_semantic_version(self.method_version)
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.implementation_sha256 != INTERVAL_PROPERTY_COMPARISON_IMPLEMENTATION_SHA256:
            raise ValueError("property-comparison spec names another implementation")
        if (
            self.method_key != INTERVAL_PROPERTY_COMPARISON_METHOD_KEY
            or self.method_version != INTERVAL_PROPERTY_COMPARISON_METHOD_VERSION
        ):
            raise ValueError("property-comparison spec names another registered method")
        if self.supported_comparison_kinds != _KINDS:
            raise ValueError("property-comparison spec must bind the exact eight-kind roster")
        expected_ids = {
            "algorithm_ids": INTERVAL_PROPERTY_COMPARISON_ALGORITHM_IDS,
            "uncertainty_rule_ids": INTERVAL_PROPERTY_COMPARISON_UNCERTAINTY_RULE_IDS,
            "tolerance_ids": INTERVAL_PROPERTY_COMPARISON_TOLERANCE_IDS,
            "decisive_falsifier_ids": INTERVAL_PROPERTY_COMPARISON_DECISIVE_FALSIFIER_IDS,
        }
        for name, expected in expected_ids.items():
            values = getattr(self, name)
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
            if values != expected:
                raise ValueError(f"{name} differs from the registered method")


@dataclass(frozen=True, slots=True)
class IntervalPropertyComparisonOperands(CanonicalRecord):
    """Authenticated source/target operands; deliberately contains no verdict."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/interval-property-comparison-operands'

    operands_id: str
    comparison_kind: PropertyComparisonKind
    target_member_id: str
    property_id: str
    receiver_action_face_id: str
    predictive_level: str
    source_law_or_property: ObjectIdentity
    target_law_or_observation: ObjectIdentity
    source_operand_schema: str
    target_operand_schema: str
    domain_cell_ids: tuple[str, ...]
    excluded_field_ids: tuple[str, ...]
    compatibility_map_ids: tuple[str, ...]
    native_units: tuple[str, ...]
    native_frames: tuple[str, ...]
    source_categorical_values: tuple[str, ...]
    target_categorical_values: tuple[str, ...]
    source_metric_values: tuple[NamedDecimal, ...]
    target_metric_values: tuple[NamedDecimal, ...]
    source_uncertainty_values: tuple[NamedDecimal, ...]
    target_uncertainty_values: tuple[NamedDecimal, ...]
    source_lineage_axis_ids: tuple[str, ...]
    target_lineage_axis_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "operands_id",
            "target_member_id",
            "property_id",
            "receiver_action_face_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_schema(self.source_operand_schema)
        validate_schema(self.target_operand_schema)
        for name in (
            "domain_cell_ids",
            "excluded_field_ids",
            "compatibility_map_ids",
            "source_categorical_values",
            "target_categorical_values",
            "source_lineage_axis_ids",
            "target_lineage_axis_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if not self.domain_cell_ids or not self.compatibility_map_ids:
            raise ValueError("property comparison requires a declared domain and compatibility map")
        if not self.native_units or not self.native_frames:
            raise ValueError("property comparison requires native units and frames")
        require_sorted_unique_strings(self.native_units, field_name="native_units")
        require_sorted_unique_strings(self.native_frames, field_name="native_frames")
        if self.predictive_level not in {"CATEGORICAL", "METRIC", "DYNAMICAL"}:
            raise ValueError("property comparison has an unknown predictive level")
        require_sorted_unique_ids(
            self.source_metric_values,
            attribute="value_id",
            field_name="source_metric_values",
        )
        require_sorted_unique_ids(
            self.target_metric_values,
            attribute="value_id",
            field_name="target_metric_values",
        )
        require_sorted_unique_ids(
            self.source_uncertainty_values,
            attribute="value_id",
            field_name="source_uncertainty_values",
        )
        require_sorted_unique_ids(
            self.target_uncertainty_values,
            attribute="value_id",
            field_name="target_uncertainty_values",
        )

    @property
    def face_key(self) -> tuple[str, str, str, str]:
        return (
            self.target_member_id,
            self.property_id,
            self.receiver_action_face_id,
            self.predictive_level,
        )


@dataclass(frozen=True, slots=True)
class IntervalPropertyComparisonResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/interval-property-comparison-result'

    result_id: str
    method_spec: ObjectIdentity
    operands: ObjectIdentity
    comparison_kind: PropertyComparisonKind
    face_key: tuple[str, str, str, str]
    source_summary_values: tuple[str, ...]
    target_summary_values: tuple[str, ...]
    metric_differences: tuple[NamedDecimal, ...]
    property_preserved: bool | None
    achieved_dependence_class: str
    disposition: ScientificStatus
    decisive_falsifier_id: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if len(self.face_key) != 4:
            raise ValueError("property-comparison result lacks its complete face key")
        for value in self.face_key[:3]:
            validate_stable_id(value, field_name="face_key")
        if self.face_key[3] not in {"CATEGORICAL", "METRIC", "DYNAMICAL"}:
            raise ValueError("property-comparison result has an unknown predictive level")
        require_sorted_unique_strings(
            self.source_summary_values, field_name="source_summary_values"
        )
        require_sorted_unique_strings(
            self.target_summary_values, field_name="target_summary_values"
        )
        require_sorted_unique_ids(
            self.metric_differences,
            attribute="value_id",
            field_name="metric_differences",
        )
        validate_stable_id(
            self.achieved_dependence_class,
            field_name="achieved_dependence_class",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is ScientificStatus.SUPPORTED:
            if self.property_preserved is not True or self.decisive_falsifier_id is not None:
                raise ValueError("supported property result is inconsistent")
        elif self.disposition is ScientificStatus.NOT_SUPPORTED:
            if self.property_preserved is not False or self.decisive_falsifier_id is None:
                raise ValueError("opposed property result requires its decisive falsifier")
        elif self.disposition is ScientificStatus.UNEVALUABLE:
            if self.property_preserved is not None or self.decisive_falsifier_id is not None:
                raise ValueError("unevaluable property result fabricates a verdict")
        else:
            raise ValueError("property comparison has an unsupported disposition")
        if self.decisive_falsifier_id is not None:
            validate_stable_id(self.decisive_falsifier_id, field_name="decisive_falsifier_id")


def _metric_map(values: tuple[NamedDecimal, ...]) -> dict[str, NamedDecimal]:
    return {value.value_id: value for value in values}


def _uncertainty_map(values: tuple[NamedDecimal, ...]) -> dict[str, Decimal]:
    return {value.value_id: value.value for value in values}


def _dependence(operands: IntervalPropertyComparisonOperands) -> str:
    source = set(operands.source_lineage_axis_ids)
    target = set(operands.target_lineage_axis_ids)
    if not source or not target:
        return "dependence.unresolved-lineage"
    if source == target:
        return "dependence.shared-lineage"
    if source.isdisjoint(target):
        return "dependence.disjoint-declared-lineage"
    return "dependence.partially-shared-lineage"


def _sign(value: Decimal, uncertainty: Decimal, tolerance: Decimal) -> int | None:
    lower = value - uncertainty - tolerance
    upper = value + uncertainty + tolerance
    if lower > 0:
        return 1
    if upper < 0:
        return -1
    if lower == upper == 0:
        return 0
    return None


def _compare(
    operands: IntervalPropertyComparisonOperands,
) -> tuple[bool | None, tuple[NamedDecimal, ...]]:
    tolerance = _METHOD_OWNED_TOLERANCE_BY_KIND[operands.comparison_kind]
    source = _metric_map(operands.source_metric_values)
    target = _metric_map(operands.target_metric_values)
    source_uncertainty = _uncertainty_map(operands.source_uncertainty_values)
    target_uncertainty = _uncertainty_map(operands.target_uncertainty_values)
    common = tuple(sorted(set(source) & set(target)))
    differences = tuple(
        NamedDecimal(
            value_id=f"difference.{value_id}",
            value=target[value_id].value - source[value_id].value,
            unit=source[value_id].unit,
        )
        for value_id in common
        if source[value_id].unit == target[value_id].unit
    )
    categorical = (
        bool(operands.source_categorical_values)
        and operands.source_categorical_values == operands.target_categorical_values
    )
    kind = operands.comparison_kind
    if kind in {
        PropertyComparisonKind.RECEIVER_CONFLICT,
        PropertyComparisonKind.REACHABILITY,
        PropertyComparisonKind.BRANCH_FRAGILITY,
    }:
        if not operands.source_categorical_values or not operands.target_categorical_values:
            return None, differences
        return categorical, differences
    if not common or len(differences) != len(common):
        return None, differences
    if any(
        value_id not in source_uncertainty or value_id not in target_uncertainty
        for value_id in common
    ):
        return None, differences
    signs = tuple(
        (
            _sign(source[value_id].value, source_uncertainty[value_id], tolerance),
            _sign(target[value_id].value, target_uncertainty[value_id], tolerance),
        )
        for value_id in common
    )
    if kind in {
        PropertyComparisonKind.SIGN,
        PropertyComparisonKind.HISTORY_DEPENDENCE,
    }:
        if any(left is None or right is None for left, right in signs):
            return None, differences
        return all(left == right for left, right in signs), differences
    if kind is PropertyComparisonKind.ORDER:
        if len(common) < 2:
            return None, differences
        pairs = tuple(
            (left, right) for index, left in enumerate(common) for right in common[index + 1 :]
        )
        source_order = tuple(
            _sign(
                source[right].value - source[left].value,
                source_uncertainty[right] + source_uncertainty[left],
                tolerance,
            )
            for left, right in pairs
        )
        target_order = tuple(
            _sign(
                target[right].value - target[left].value,
                target_uncertainty[right] + target_uncertainty[left],
                tolerance,
            )
            for left, right in pairs
        )
        if None in source_order or None in target_order:
            return None, differences
        return source_order == target_order, differences
    if kind is PropertyComparisonKind.RANK_NULL:
        if any(left is None or right is None for left, right in signs):
            return None, differences
        return tuple(left == 0 for left, _ in signs) == tuple(
            right == 0 for _, right in signs
        ), differences
    if kind is PropertyComparisonKind.SAFETY_CONSTRAINT:
        if any(left is None or right is None for left, right in signs):
            return None, differences
        defined_signs = tuple((cast(int, left), cast(int, right)) for left, right in signs)
        return all((left >= 0) == (right >= 0) for left, right in defined_signs), differences
    return None, differences


def evaluate_property_comparison(
    *,
    spec: IntervalPropertyComparisonMethodSpec,
    operands: IntervalPropertyComparisonOperands,
) -> IntervalPropertyComparisonResult:
    """Compute the preservation result; callers cannot supply or toggle it."""

    kind_index = spec.supported_comparison_kinds.index(operands.comparison_kind)
    preserved, differences = _compare(operands)
    if preserved is True:
        disposition = ScientificStatus.SUPPORTED
        falsifier = None
        reasons = ("PROPERTY_COMPARISON_SUPPORTED",)
    elif preserved is False:
        disposition = ScientificStatus.NOT_SUPPORTED
        falsifier = spec.decisive_falsifier_ids[kind_index]
        reasons = ("PROPERTY_COMPARISON_DECISIVE_FALSIFIER",)
    else:
        disposition = ScientificStatus.UNEVALUABLE
        falsifier = None
        reasons = ("PROPERTY_COMPARISON_OPERANDS_UNEVALUABLE",)
    spec_identity = ObjectIdentity.from_record(spec.spec_id, spec)
    operands_identity = ObjectIdentity.from_record(operands.operands_id, operands)
    identity_seed = {
        "spec": spec_identity,
        "operands": operands_identity,
        "preserved": preserved,
        "disposition": disposition,
    }
    result_id = (
        f"property-comparison-result.{sha256(canonical_json_bytes(identity_seed)).hexdigest()[:32]}"
    )
    return IntervalPropertyComparisonResult(
        result_id=result_id,
        method_spec=spec_identity,
        operands=operands_identity,
        comparison_kind=operands.comparison_kind,
        face_key=operands.face_key,
        source_summary_values=operands.source_categorical_values,
        target_summary_values=operands.target_categorical_values,
        metric_differences=differences,
        property_preserved=preserved,
        achieved_dependence_class=_dependence(operands),
        disposition=disposition,
        decisive_falsifier_id=falsifier,
        reason_codes=reasons,
    )


__all__ = [
    'INTERVAL_PROPERTY_COMPARISON_ALGORITHM_IDS',
    'INTERVAL_PROPERTY_COMPARISON_DECISIVE_FALSIFIER_IDS',
    'INTERVAL_PROPERTY_COMPARISON_IMPLEMENTATION_SHA256',
    'INTERVAL_PROPERTY_COMPARISON_METHOD_KEY',
    'INTERVAL_PROPERTY_COMPARISON_METHOD_VERSION',
    'INTERVAL_PROPERTY_COMPARISON_TOLERANCE_IDS',
    'INTERVAL_PROPERTY_COMPARISON_UNCERTAINTY_RULE_IDS',
    'PropertyComparisonKind',
    'IntervalPropertyComparisonMethodSpec',
    'IntervalPropertyComparisonOperands',
    'IntervalPropertyComparisonResult',
    'evaluate_property_comparison',
]
