"""Outcome-blind NREL physical-unit mapping and power unit, mapping and power design surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateComparatorKind, IndependentSubstrateDenominatorCandidate, IndependentSubstrateMappingConstructor, IndependentSubstrateRoleBinding, IndependentSubstrateScientificDesignBasis, IndependentSubstrateTargetMappingCandidate, IndependentSubstrateTargetKind
from empirical_lawhood.adapters.methods.structured_target import IndependentSubstrateStructuredPowerFreeze, IndependentSubstrateStructuredTargetDesignFreeze
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .contracts import NRELArchiveSourceBinding


NREL_NATIVE_ACTION_IDS = (
    "frequency-setpoint",
    "hold",
    "load-condition",
    "voltage-setpoint",
)
NREL_RECEIVER_GAUGE_IDS = ("ac-electrical", "dc-electrical")
NREL_HORIZON_TICKS = (1, 2)
NREL_OBSERVATION_RESOLUTION_IDS = (
    "preparation-aggregate",
    "sample-native",
)
NREL_SIMULTANEOUS_FAMILY_IDS = (
    "all-decisive-falsifiers",
    "all-primary-estimands",
    "all-topology-metric-exchanges",
)


@dataclass(frozen=True, slots=True)
class NRELPhysicalUnitInventory(CanonicalRecord):
    """physical-unit grouping grouping/split decision made without receiver outcomes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/nrel-inverter-archive/nrel-physical-unit-inventory'

    inventory_id: str
    source_binding: ObjectIdentity
    apparatus_id: str
    site_id: str
    preparation_ids: tuple[str, ...]
    development_preparation_ids: tuple[str, ...]
    evaluation_preparation_ids: tuple[str, ...]
    unit_hierarchy_resolved: bool
    reset_identity_resolved: bool
    rows_count_as_units: bool
    receiver_values_accessed_during_inventory: bool
    split_committed_before_receiver_decode: bool
    prospective_prediction_eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("inventory_id", "apparatus_id", "site_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.source_binding.object_schema != NRELArchiveSourceBinding.SCHEMA:
            raise ValueError("NREL unit inventory source identity differs")
        for name in (
            "preparation_ids",
            "development_preparation_ids",
            "evaluation_preparation_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        development = set(self.development_preparation_ids)
        evaluation = set(self.evaluation_preparation_ids)
        if development & evaluation or development | evaluation != set(self.preparation_ids):
            raise ValueError("NREL development/evaluation unit partition differs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        eligible = all(
            (
                self.unit_hierarchy_resolved,
                self.reset_identity_resolved,
                not self.rows_count_as_units,
                not self.receiver_values_accessed_during_inventory,
                self.split_committed_before_receiver_decode,
            )
        )
        if self.prospective_prediction_eligible != eligible:
            raise ValueError("NREL prediction eligibility differs from physical-unit grouping facts")
        if eligible and self.reason_codes:
            raise ValueError("eligible NREL unit inventory cannot retain stop reasons")
        if not eligible and not self.reason_codes:
            raise ValueError("ineligible NREL unit inventory requires a typed reason")


def _role_bindings(
    *,
    token: str,
    receiver_constructor: IndependentSubstrateMappingConstructor,
    receiver_ids: tuple[str, ...],
) -> tuple[IndependentSubstrateRoleBinding, ...]:
    specifications = (
        (
            "A",
            (
                "accepted-native-setpoint",
                "applied-native-setpoint",
                "realized-native-setpoint",
                "requested-native-setpoint",
            ),
            IndependentSubstrateMappingConstructor.DIRECT_NATIVE_ROLE_BINDING,
        ),
        (
            "D",
            ("apparatus", "load-condition", "preparation-reset", "site"),
            IndependentSubstrateMappingConstructor.DECLARED_TARGET_NATIVE_AGGREGATE,
        ),
        (
            "H",
            ("prior-logged-interventions", "preparation-start-state"),
            IndependentSubstrateMappingConstructor.DECLARED_TARGET_NATIVE_RESTRICTION,
        ),
        ("R", receiver_ids, receiver_constructor),
        (
            "tau",
            ("native-sample-lag-000001", "native-sample-lag-000002"),
            IndependentSubstrateMappingConstructor.DECLARED_HORIZON_SELECTION,
        ),
    )
    return tuple(
        IndependentSubstrateRoleBinding(
            binding_id=f"binding.nrel.{index:02d}-{role.lower()}.{token}",
            structural_recurrence_role_id=role,
            target_native_role_ids=tuple(sorted(native_ids)),
            constructor=constructor,
            native_units_preserved=True,
            native_frame_preserved=True,
            receiver_direction_preserved=True,
            causal_cutoff_preserved=True,
            outcome_derived=False,
        )
        for index, (role, native_ids, constructor) in enumerate(specifications)
    )


def nrel_mapping_candidates() -> tuple[IndependentSubstrateTargetMappingCandidate, ...]:
    roster_id = "roster.independent-substrate-grounding.nrel.mapping-candidates"
    candidates = (
        IndependentSubstrateTargetMappingCandidate(
            candidate_id="mapping.nrel.00-native-ac-dc",
            target_slot=IndependentSubstrateTargetKind.NREL_INVERTER,
            finite_roster_id=roster_id,
            role_bindings=_role_bindings(
                token="native-ac-dc",
                receiver_constructor=IndependentSubstrateMappingConstructor.DIRECT_NATIVE_ROLE_BINDING,
                receiver_ids=("ac-power-w", "dc-power-w"),
            ),
            requested_accepted_applied_realized_distinct=True,
            target_native_action_chart_preserved=True,
            development_authored=True,
        ),
        IndependentSubstrateTargetMappingCandidate(
            candidate_id="mapping.nrel.01-conversion-gauge",
            target_slot=IndependentSubstrateTargetKind.NREL_INVERTER,
            finite_roster_id=roster_id,
            role_bindings=_role_bindings(
                token="conversion-gauge",
                receiver_constructor=(IndependentSubstrateMappingConstructor.DECLARED_RECEIVER_GAUGE_PROJECTION),
                receiver_ids=("ac-power-w", "dc-to-ac-conversion-gauge"),
            ),
            requested_accepted_applied_realized_distinct=True,
            target_native_action_chart_preserved=True,
            development_authored=True,
        ),
    )
    return tuple(sorted(candidates, key=lambda value: value.candidate_id))


def nrel_denominator_candidates() -> tuple[IndependentSubstrateDenominatorCandidate, ...]:
    candidates = (
        IndependentSubstrateDenominatorCandidate(
            candidate_id="denominator.nrel.00-proposed",
            factor_ids=("apparatus", "load-condition", "preparation-reset", "site"),
            stratum_ids=("run",),
            impossible=False,
            impossible_reason=None,
        ),
        IndependentSubstrateDenominatorCandidate(
            candidate_id="denominator.nrel.01-setpoint-family-split",
            factor_ids=("apparatus", "load-condition", "preparation-reset", "site"),
            stratum_ids=("run", "setpoint-family"),
            impossible=False,
            impossible_reason=None,
        ),
        IndependentSubstrateDenominatorCandidate(
            candidate_id="denominator.nrel.02-run-merged",
            factor_ids=("apparatus", "load-condition", "site"),
            stratum_ids=("run",),
            impossible=False,
            impossible_reason=None,
        ),
        IndependentSubstrateDenominatorCandidate(
            candidate_id="denominator.nrel.03-reset-omitted",
            factor_ids=("apparatus", "load-condition", "site"),
            stratum_ids=(),
            impossible=True,
            impossible_reason="physical preparation/reset identity cannot be omitted",
        ),
    )
    return tuple(sorted(candidates, key=lambda value: value.candidate_id))


def build_nrel_target_design(
    *,
    source: NRELArchiveSourceBinding,
    common_design_basis: ObjectIdentity,
    minimum_development_units: int,
    minimum_evaluation_units: int,
) -> IndependentSubstrateStructuredTargetDesignFreeze:
    """Instantiate predictive predictive design and power freeze only for a controlled public archive split."""

    if common_design_basis.object_schema != IndependentSubstrateScientificDesignBasis.SCHEMA:
        raise ValueError("NREL common scientific design basis differs")
    if not source.prospective_prediction_eligible:
        raise ValueError("NREL uncontrolled public outcomes permit descriptive analysis only")
    return IndependentSubstrateStructuredTargetDesignFreeze(
        design_id="design.independent-substrate-grounding.nrel.predictive-unit-mapping-design",
        slot=IndependentSubstrateTargetKind.NREL_INVERTER,
        source_binding=ObjectIdentity.from_record(source.binding_id, source),
        common_design_basis=common_design_basis,
        mapping_candidates=nrel_mapping_candidates(),
        denominator_candidates=nrel_denominator_candidates(),
        native_action_ids=NREL_NATIVE_ACTION_IDS,
        receiver_gauge_ids=NREL_RECEIVER_GAUGE_IDS,
        horizon_ticks=NREL_HORIZON_TICKS,
        graph_or_observation_resolution_ids=NREL_OBSERVATION_RESOLUTION_IDS,
        comparator_kinds=tuple(sorted(IndependentSubstrateComparatorKind, key=lambda value: value.value)),
        minimum_development_units=minimum_development_units,
        minimum_evaluation_units=minimum_evaluation_units,
        minimum_prospective_validation_units=0,
        physical_response_margin_rule=(
            "NATIVE_AC_DC_RESPONSE_DISTANCE_WITH_CALIBRATION_UNCERTAINTY"
        ),
        certification_margin_rule=(
            "SIMULTANEOUS_COMPLETE_PREPARATION_INTERVAL_OVER_PRIMARY_FAMILIES"
        ),
        complete_unit_resampling=True,
        frozen_before_development=True,
        protected_outcome_access_count=0,
    )


def freeze_nrel_power(
    *,
    source: NRELArchiveSourceBinding,
    design: IndependentSubstrateStructuredTargetDesignFreeze,
    inventory: NRELPhysicalUnitInventory,
    simultaneous_precision_passed: bool,
) -> IndependentSubstrateStructuredPowerFreeze:
    if (
        design.slot is not IndependentSubstrateTargetKind.NREL_INVERTER
        or design.source_binding != ObjectIdentity.from_record(source.binding_id, source)
        or inventory.source_binding != design.source_binding
        or inventory.apparatus_id != source.apparatus_id
        or inventory.site_id != source.site_id
    ):
        raise ValueError("NREL power freeze source/design/inventory differs")
    if inventory.prospective_prediction_eligible != source.prospective_prediction_eligible:
        raise ValueError("NREL unit inventory visibility eligibility differs from source")
    count_passed = (
        len(inventory.development_preparation_ids) >= design.minimum_development_units
        and len(inventory.evaluation_preparation_ids) >= design.minimum_evaluation_units
    )
    joint_passed = count_passed and simultaneous_precision_passed
    reasons: list[str] = []
    if not count_passed:
        reasons.append("NREL_PREPARATION_PANEL_BELOW_FROZEN_MINIMUM")
    if not simultaneous_precision_passed:
        reasons.append("NREL_JOINT_PANEL_GRID_RECEIVER_PRECISION_NOT_ESTABLISHED")
    return IndependentSubstrateStructuredPowerFreeze(
        freeze_id="power-freeze.independent-substrate-grounding.nrel.predictive-unit-power-freeze",
        design=ObjectIdentity.from_record(design.design_id, design),
        development_unit_ids=inventory.development_preparation_ids,
        evaluation_unit_ids=inventory.evaluation_preparation_ids,
        prospective_validation_unit_ids=(),
        simultaneous_family_ids=NREL_SIMULTANEOUS_FAMILY_IDS,
        joint_precision_passed=joint_passed,
        panel_envelope_limited=not joint_passed,
        reason_codes=tuple(sorted(reasons)),
    )


__all__ = [
    'NRELPhysicalUnitInventory',
    "NREL_HORIZON_TICKS",
    "NREL_NATIVE_ACTION_IDS",
    "NREL_OBSERVATION_RESOLUTION_IDS",
    "NREL_RECEIVER_GAUGE_IDS",
    "NREL_SIMULTANEOUS_FAMILY_IDS",
    "build_nrel_target_design",
    "freeze_nrel_power",
    "nrel_denominator_candidates",
    "nrel_mapping_candidates",
]
