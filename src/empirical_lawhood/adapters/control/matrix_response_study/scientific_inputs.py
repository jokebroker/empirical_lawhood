"""Original numerical allocations with current policy and selection custody.

These inputs require externally verified original exports. They do not verify
those exports, authorize native access, or carry forward scientific qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar, TYPE_CHECKING

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes, validate_sha256

if TYPE_CHECKING:
    from .transient_policy import (
        MatrixResponseTransientControlledInvarianceFrozenInterventionRule,
        MatrixResponseTransientControlledInvariancePhaseFeature,
        MatrixResponseTransientControlledInvariancePhaseRule,
    )


def policy_allocation_context_sha256(
    rule: MatrixResponseTransientControlledInvarianceFrozenInterventionRule,
    feature_series: tuple[tuple[MatrixResponseTransientControlledInvariancePhaseFeature, ...], ...],
) -> str:
    return sha256(canonical_json_bytes((
        rule.fingerprint(),
        tuple(tuple(feature.fingerprint() for feature in prefix) for prefix in feature_series),
    ))).hexdigest()


def phase_feature_context_sha256(
    features: tuple[MatrixResponseTransientControlledInvariancePhaseFeature, ...],
) -> str:
    return sha256(canonical_json_bytes(tuple(feature.fingerprint() for feature in features))).hexdigest()


def phase_rule_scientific_key_sha256(
    rule: MatrixResponseTransientControlledInvariancePhaseRule,
) -> str:
    return sha256(canonical_json_bytes(tuple(
        (predicate.feature_name, predicate.direction, predicate.threshold)
        for predicate in rule.predicates
    ))).hexdigest()


def _require_export_custody(original_source: ObjectIdentity, export_receipt: ObjectIdentity) -> None:
    if not isinstance(original_source, ObjectIdentity) or not isinstance(export_receipt, ObjectIdentity):
        raise ValueError("numerical inputs require original source and external export custody")


@dataclass(frozen=True, slots=True)
class MatrixResponsePolicyAllocationScientificInputs(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/scientific-input/matrix-response-policy-allocation"

    original_source: ObjectIdentity
    export_receipt: ObjectIdentity
    source_original_context_sha256: str
    current_context_sha256: str
    open_loop_seed_sha256: str
    stratum_seed_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_export_custody(self.original_source, self.export_receipt)
        for name in ("source_original_context_sha256", "current_context_sha256", "open_loop_seed_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if not isinstance(self.stratum_seed_sha256s, tuple) or len(self.stratum_seed_sha256s) != 4:
            raise ValueError("policy allocation requires all four ordered amplitude strata")
        for seed in self.stratum_seed_sha256s:
            validate_sha256(seed, field_name="stratum_seed_sha256")
        prefixes = tuple(seed[:32] for seed in (self.open_loop_seed_sha256, *self.stratum_seed_sha256s))
        if len(set(prefixes)) != 5:
            raise ValueError("policy allocation repeats a PCG allocation")


@dataclass(frozen=True, slots=True)
class MatrixResponsePhaseRuleSelectionScientificInputs(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/scientific-input/matrix-response-phase-rule-selection"

    original_source: ObjectIdentity
    export_receipt: ObjectIdentity
    source_original_context_sha256: str
    current_config_sha256: str
    current_features_sha256: str
    comparison_order: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        _require_export_custody(self.original_source, self.export_receipt)
        for name in ("source_original_context_sha256", "current_config_sha256", "current_features_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if not isinstance(self.comparison_order, tuple) or len(self.comparison_order) > 36792:
            raise ValueError("phase selection requires the complete bounded comparison order")
        keys: list[str] = []
        ranks: list[int] = []
        for pair in self.comparison_order:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("phase selection requires scientific key and original numeric rank pairs")
            key, rank = pair
            validate_sha256(key, field_name="scientific_rule_key_sha256")
            if type(rank) is not int or not 0 <= rank < 2**64:
                raise ValueError("phase selection rank is outside its original unsigned allocation")
            keys.append(key)
            ranks.append(rank)
        if tuple(keys) != tuple(sorted(set(keys))) or len(set(ranks)) != len(ranks):
            raise ValueError("phase selection keys and numerical ranks must be complete and unique")


def require_policy_allocation_scientific_inputs(
    value: MatrixResponsePolicyAllocationScientificInputs, *,
    rule: MatrixResponseTransientControlledInvarianceFrozenInterventionRule,
    feature_series: tuple[tuple[MatrixResponseTransientControlledInvariancePhaseFeature, ...], ...],
) -> MatrixResponsePolicyAllocationScientificInputs:
    if not isinstance(value, MatrixResponsePolicyAllocationScientificInputs):
        raise ValueError("policy allocation requires complete typed original numerical inputs")
    if value.current_context_sha256 != policy_allocation_context_sha256(rule, feature_series):
        raise ValueError("policy allocation inputs bind another current rule or ordered feature prefixes")
    return value


def require_phase_rule_selection_scientific_inputs(
    value: MatrixResponsePhaseRuleSelectionScientificInputs, *,
    current_config_sha256: str,
    features: tuple[MatrixResponseTransientControlledInvariancePhaseFeature, ...],
    complete_rules: tuple[MatrixResponseTransientControlledInvariancePhaseRule, ...],
) -> dict[str, int]:
    if not isinstance(value, MatrixResponsePhaseRuleSelectionScientificInputs):
        raise ValueError("phase selection requires complete typed original numerical comparison inputs")
    if (value.current_config_sha256 != current_config_sha256
        or value.current_features_sha256 != phase_feature_context_sha256(features)):
        raise ValueError("phase selection inputs bind another current configuration or feature acquisition order")
    complete_keys = tuple(sorted(phase_rule_scientific_key_sha256(rule) for rule in complete_rules))
    if tuple(key for key, _ in value.comparison_order) != complete_keys:
        raise ValueError("phase selection inputs do not cover the complete current scientific predicate roster")
    return dict(value.comparison_order)

