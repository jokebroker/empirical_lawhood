"""Causal D/H/R construction with one strict pre-action cutoff."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)

from .contracts import (
    FairMastMaterializationReceipt,
    MastCausalState,
    MastEvent,
    MastEventDisposition,
    MastFeatureRole,
    MastFeatureValue,
    MastShotAssignment,
    MastShotRole,
)


_FORBIDDEN_FIELD_FRAGMENTS = ("endpoint", "final-action", "outcome", "post-action")


@dataclass(frozen=True, slots=True)
class MastPreActionField(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/mast-pre-action-field'

    field_id: str
    source_signal_id: str
    role: MastFeatureRole
    coordinate_s: Decimal
    values: tuple[Decimal, ...]
    radial_coordinates: tuple[Decimal, ...]
    native_unit: str
    missing_mask: tuple[bool, ...]
    quality_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.field_id, field_name="field_id")
        validate_stable_id(self.source_signal_id, field_name="source_signal_id")
        validate_decimal(self.coordinate_s, field_name="coordinate_s")
        validate_nonempty(self.native_unit, field_name="native_unit")
        if any(fragment in self.source_signal_id for fragment in _FORBIDDEN_FIELD_FRAGMENTS):
            raise ValueError("post-action/action-label/outcome field is forbidden")
        if not self.values or len(self.values) != len(self.missing_mask):
            raise ValueError("pre-action field values require a missingness mask")
        if self.radial_coordinates and len(self.radial_coordinates) != len(self.values):
            raise ValueError("pre-action radial coordinates differ from values")
        for value in (*self.values, *self.radial_coordinates):
            validate_decimal(value, field_name="field_value")
        require_sorted_unique_strings(self.quality_reason_codes, field_name="quality_reason_codes")


@dataclass(frozen=True, slots=True)
class MastFeatureRule(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/mast-feature-rule'

    rule_id: str
    cutoff_rule_id: str
    permitted_field_ids: tuple[str, ...]
    transform_id: str
    transform_fit_role: MastShotRole
    lookback_s: Decimal
    development_only: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("rule_id", self.rule_id),
            ("cutoff_rule_id", self.cutoff_rule_id),
            ("transform_id", self.transform_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.permitted_field_ids, field_name="permitted_field_ids", allow_empty=False
        )
        validate_decimal(self.lookback_s, field_name="lookback_s", minimum=Decimal(0))
        if self.lookback_s == 0:
            raise ValueError("feature lookback must be positive")
        if self.transform_fit_role is not MastShotRole.DEVELOPMENT:
            raise ValueError("feature transforms may be fitted only on development shots")
        if not self.development_only:
            raise ValueError("Phase 4 feature rule is development-only and unissued")


def decode_mast_feature_rule(payload: bytes) -> MastFeatureRule:
    return decode_canonical_bytes(payload, MastFeatureRule, maximum_bytes=64 * 1024)


def build_mast_causal_state(
    *,
    assignment: MastShotAssignment,
    event: MastEvent,
    fields: tuple[MastPreActionField, ...],
    materialization: FairMastMaterializationReceipt,
    rule: MastFeatureRule,
) -> MastCausalState:
    if event.disposition not in {MastEventDisposition.ACCEPTED, MastEventDisposition.CANDIDATE}:
        raise ValueError("causal state requires an eligible event")
    if event.t0_s is None:
        raise ValueError("eligible event has no t0")
    if assignment.shot_id != event.shot_id or event.shot_id != materialization.shot_id:
        raise ValueError("MAST shot identity differs across source/event/assignment")
    if assignment.campaign is not event.campaign or event.campaign is not materialization.campaign:
        raise ValueError("MAST campaign identity differs")
    require_sorted_unique_ids(fields, attribute="field_id", field_name="fields")
    if tuple(value.field_id for value in fields) != rule.permitted_field_ids:
        raise ValueError("feature inputs differ from the closed allowlist")
    lower = event.t0_s - rule.lookback_s
    if any(not lower <= value.coordinate_s < event.t0_s for value in fields):
        raise ValueError("feature input violates the single pre-t0 cutoff")
    if {value.role for value in fields} != set(MastFeatureRole):
        raise ValueError("feature inputs must expose D, H and R distinctly")
    features = tuple(
        MastFeatureValue(
            feature_id=value.field_id,
            source_signal_id=value.source_signal_id,
            role=value.role,
            values=value.values,
            radial_coordinates=value.radial_coordinates,
            native_unit=value.native_unit,
            missing_mask=value.missing_mask,
            quality_reason_codes=value.quality_reason_codes,
            latest_source_coordinate_s=value.coordinate_s,
            transform_id=rule.transform_id,
        )
        for value in fields
    )
    return MastCausalState(
        state_id=f"mast-state-{assignment.shot_id}",
        shot_id=assignment.shot_id,
        campaign=assignment.campaign,
        role=assignment.role,
        t0_s=event.t0_s,
        cutoff_rule_id=rule.cutoff_rule_id,
        features=features,
        outcome_access=(
            OutcomeAccess.OUTCOME_BLIND
            if assignment.role is MastShotRole.PROTECTED_EVALUATION
            else OutcomeAccess.DEVELOPMENT_VISIBLE
        ),
        source_materialization=ObjectIdentity.from_record(
            materialization.receipt_id, materialization
        ),
    )


__all__ = [
    "MastFeatureRule",
    "MastPreActionField",
    "build_mast_causal_state",
    "decode_mast_feature_rule",
]
