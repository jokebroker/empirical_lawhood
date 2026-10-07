"""Outcome-blind Grid2Op action-receiver design/complete-chronic power freeze target design construction.

The builders in this module instantiate the common outcome-blind scientific grammar grammar using only an
exact held source binding and predeclared native Grid2Op semantics.  They do
not import Grid2Op, inspect simulator outcomes, select a mapping, or issue an
experiment.
"""

from __future__ import annotations

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateComparatorKind, IndependentSubstrateDenominatorCandidate, IndependentSubstrateMappingConstructor, IndependentSubstrateRoleBinding, IndependentSubstrateScientificDesignBasis, IndependentSubstrateTargetMappingCandidate, IndependentSubstrateTargetKind
from empirical_lawhood.adapters.methods.structured_target import IndependentSubstrateStructuredPowerFreeze, IndependentSubstrateStructuredTargetDesignFreeze
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .contracts import Grid2OpSourceBinding


GRID2OP_SOURCE_SUPPORTED_ACTION_KIND_IDS = (
    "change-bus",
    "hold",
    "redispatch",
    "set-bus",
    "set-line-status",
)
# The frozen independent substrate grounding action chart is deliberately narrower than every operation
# the source can express.  It is a finite measured fibre: two atomic line
# disconnections, their predeclared joint action, and mandatory hold.
GRID2OP_NATIVE_ACTION_IDS = (
    "disconnect-line-000",
    "disconnect-line-001",
    "disconnect-lines-000-001",
    "hold",
)
GRID2OP_RECEIVER_GAUGE_IDS = ("thermal", "topology")
GRID2OP_HORIZON_TICKS = (1, 3, 6)
GRID2OP_GRAPH_RESOLUTION_IDS = ("graph-busbar", "graph-substation")
GRID2OP_MINIMUM_DEVELOPMENT_UNITS = 18
GRID2OP_MINIMUM_EVALUATION_UNITS = 18
GRID2OP_MINIMUM_PROSPECTIVE_VALIDATION_UNITS = 12
GRID2OP_SIMULTANEOUS_FAMILY_IDS = (
    "all-decisive-falsifiers",
    "all-primary-estimands",
    "all-topology-metric-exchanges",
)


def _role_bindings(
    *,
    candidate_token: str,
    receiver_constructor: IndependentSubstrateMappingConstructor,
    receiver_native_ids: tuple[str, ...],
    native_action_ids: tuple[str, ...],
) -> tuple[IndependentSubstrateRoleBinding, ...]:
    specifications = (
        (
            "A",
            native_action_ids,
            IndependentSubstrateMappingConstructor.DIRECT_NATIVE_ROLE_BINDING,
        ),
        (
            "D",
            (
                "backend-class-and-version",
                "chronic-exogenous-event",
                "grid-and-rules",
                "observation-operator",
            ),
            IndependentSubstrateMappingConstructor.DECLARED_TARGET_NATIVE_AGGREGATE,
        ),
        (
            "H",
            (
                "initial-environment-state",
                "initial-topology",
                "prior-exogenous-events",
            ),
            IndependentSubstrateMappingConstructor.DECLARED_TARGET_NATIVE_RESTRICTION,
        ),
        ("R", receiver_native_ids, receiver_constructor),
        (
            "tau",
            tuple(f"native-step-{value:06d}" for value in GRID2OP_HORIZON_TICKS),
            IndependentSubstrateMappingConstructor.DECLARED_HORIZON_SELECTION,
        ),
    )
    return tuple(
        IndependentSubstrateRoleBinding(
            binding_id=(f"binding.grid2op.{index:02d}-{role.lower()}.{candidate_token}"),
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


def grid2op_mapping_candidates(
    *, native_action_ids: tuple[str, ...] = GRID2OP_NATIVE_ACTION_IDS
) -> tuple[IndependentSubstrateTargetMappingCandidate, ...]:
    """Return the finite direct/receiver-projection action-receiver design candidate roster."""

    roster_id = "roster.independent-substrate-grounding.grid2op.mapping-candidates"
    candidates = (
        IndependentSubstrateTargetMappingCandidate(
            candidate_id="mapping.grid2op.00-native-receivers",
            target_slot=IndependentSubstrateTargetKind.GRID2OP,
            finite_roster_id=roster_id,
            role_bindings=_role_bindings(
                candidate_token="native-receivers",
                receiver_constructor=IndependentSubstrateMappingConstructor.DIRECT_NATIVE_ROLE_BINDING,
                receiver_native_ids=(
                    "connected-component-count",
                    "maximum-rho",
                ),
                native_action_ids=native_action_ids,
            ),
            requested_accepted_applied_realized_distinct=True,
            target_native_action_chart_preserved=True,
            development_authored=True,
        ),
        IndependentSubstrateTargetMappingCandidate(
            candidate_id="mapping.grid2op.01-substation-gauge-projection",
            target_slot=IndependentSubstrateTargetKind.GRID2OP,
            finite_roster_id=roster_id,
            role_bindings=_role_bindings(
                candidate_token="substation-gauge-projection",
                receiver_constructor=(IndependentSubstrateMappingConstructor.DECLARED_RECEIVER_GAUGE_PROJECTION),
                receiver_native_ids=(
                    "busbar-to-substation-topology-projection",
                    "maximum-rho",
                ),
                native_action_ids=native_action_ids,
            ),
            requested_accepted_applied_realized_distinct=True,
            target_native_action_chart_preserved=True,
            development_authored=True,
        ),
    )
    return tuple(sorted(candidates, key=lambda value: value.candidate_id))


def grid2op_denominator_candidates() -> tuple[IndependentSubstrateDenominatorCandidate, ...]:
    """Return proposed, split, merged and causally invalid omission alternatives."""

    candidates = (
        IndependentSubstrateDenominatorCandidate(
            candidate_id="denominator.grid2op.00-proposed",
            factor_ids=(
                "backend-class-and-version",
                "chronic-exogenous-event",
                "grid-and-rules",
                "observation-operator",
            ),
            stratum_ids=("initial-topology",),
            impossible=False,
            impossible_reason=None,
        ),
        IndependentSubstrateDenominatorCandidate(
            candidate_id="denominator.grid2op.01-chronic-timestamp-split",
            factor_ids=(
                "backend-class-and-version",
                "chronic-exogenous-event",
                "grid-and-rules",
                "observation-operator",
            ),
            stratum_ids=("initial-timestamp", "initial-topology"),
            impossible=False,
            impossible_reason=None,
        ),
        IndependentSubstrateDenominatorCandidate(
            candidate_id="denominator.grid2op.02-backend-merged",
            factor_ids=(
                "chronic-exogenous-event",
                "grid-and-rules",
                "observation-operator",
            ),
            stratum_ids=("initial-topology",),
            impossible=False,
            impossible_reason=None,
        ),
        IndependentSubstrateDenominatorCandidate(
            candidate_id="denominator.grid2op.03-chronic-omitted",
            factor_ids=(
                "backend-class-and-version",
                "grid-and-rules",
                "observation-operator",
            ),
            stratum_ids=("initial-topology",),
            impossible=True,
            impossible_reason=("exogenous chronic identity is a causal denominator operand"),
        ),
    )
    return tuple(sorted(candidates, key=lambda value: value.candidate_id))


def build_grid2op_target_design(
    *,
    source: Grid2OpSourceBinding,
    common_design_basis: ObjectIdentity,
) -> IndependentSubstrateStructuredTargetDesignFreeze:
    """Instantiate action-receiver design from qualified source identity without outcomes."""

    if common_design_basis.object_schema != IndependentSubstrateScientificDesignBasis.SCHEMA:
        raise ValueError("Grid2Op common scientific design basis differs")
    if not source.fresh_target_identity_disjoint:
        raise ValueError("Grid2Op prospective design requires a disjoint held source")
    return IndependentSubstrateStructuredTargetDesignFreeze(
        design_id="design.independent-substrate-grounding.grid2op.action-receiver-design",
        slot=IndependentSubstrateTargetKind.GRID2OP,
        source_binding=ObjectIdentity.from_record(source.binding_id, source),
        common_design_basis=common_design_basis,
        mapping_candidates=grid2op_mapping_candidates(
            native_action_ids=GRID2OP_NATIVE_ACTION_IDS,
        ),
        denominator_candidates=grid2op_denominator_candidates(),
        native_action_ids=GRID2OP_NATIVE_ACTION_IDS,
        receiver_gauge_ids=GRID2OP_RECEIVER_GAUGE_IDS,
        horizon_ticks=GRID2OP_HORIZON_TICKS,
        graph_or_observation_resolution_ids=GRID2OP_GRAPH_RESOLUTION_IDS,
        comparator_kinds=tuple(sorted(IndependentSubstrateComparatorKind, key=lambda value: value.value)),
        minimum_development_units=GRID2OP_MINIMUM_DEVELOPMENT_UNITS,
        minimum_evaluation_units=GRID2OP_MINIMUM_EVALUATION_UNITS,
        minimum_prospective_validation_units=GRID2OP_MINIMUM_PROSPECTIVE_VALIDATION_UNITS,
        physical_response_margin_rule=(
            "NATIVE_THERMAL_OVERLOAD_DISTANCE_AND_TOPOLOGY_STATE_CHANGE"
        ),
        certification_margin_rule=(
            "SIMULTANEOUS_COMPLETE_CHRONIC_INTERVAL_OVER_ALL_PRIMARY_FAMILIES"
        ),
        complete_unit_resampling=True,
        frozen_before_development=True,
        protected_outcome_access_count=0,
    )


def freeze_grid2op_power(
    *,
    source: Grid2OpSourceBinding,
    design: IndependentSubstrateStructuredTargetDesignFreeze,
    development_chronic_ids: tuple[str, ...],
    evaluation_chronic_ids: tuple[str, ...],
    prospective_validation_chronic_ids: tuple[str, ...],
    simultaneous_precision_passed: bool,
) -> IndependentSubstrateStructuredPowerFreeze:
    """Freeze disjoint complete-chronic power freeze chronic rosters and a joint precision disposition."""

    if design.slot is not IndependentSubstrateTargetKind.GRID2OP or design.source_binding != (
        ObjectIdentity.from_record(source.binding_id, source)
    ):
        raise ValueError("Grid2Op power freeze source/design differs")
    available = {value.chronic_id for value in source.chronic_bindings}
    requested = set(development_chronic_ids) | set(evaluation_chronic_ids) | set(prospective_validation_chronic_ids)
    if not requested.issubset(available):
        raise ValueError("Grid2Op power roster names an unqualified chronic")
    count_passed = (
        len(development_chronic_ids) >= design.minimum_development_units
        and len(evaluation_chronic_ids) >= design.minimum_evaluation_units
        and len(prospective_validation_chronic_ids) >= design.minimum_prospective_validation_units
    )
    joint_passed = count_passed and simultaneous_precision_passed
    reasons: list[str] = []
    if not count_passed:
        reasons.append("GRID2OP_PLANNED_CHRONIC_PANEL_BELOW_FROZEN_MINIMUM")
    if not simultaneous_precision_passed:
        reasons.append("GRID2OP_JOINT_PANEL_GRID_RECEIVER_PRECISION_NOT_ESTABLISHED")
    return IndependentSubstrateStructuredPowerFreeze(
        freeze_id="power-freeze.independent-substrate-grounding.grid2op.complete-chronic-power-freeze",
        design=ObjectIdentity.from_record(design.design_id, design),
        development_unit_ids=development_chronic_ids,
        evaluation_unit_ids=evaluation_chronic_ids,
        prospective_validation_unit_ids=prospective_validation_chronic_ids,
        simultaneous_family_ids=GRID2OP_SIMULTANEOUS_FAMILY_IDS,
        joint_precision_passed=joint_passed,
        panel_envelope_limited=not joint_passed,
        reason_codes=tuple(sorted(reasons)),
    )


__all__ = [
    "GRID2OP_GRAPH_RESOLUTION_IDS",
    "GRID2OP_HORIZON_TICKS",
    "GRID2OP_MINIMUM_DEVELOPMENT_UNITS",
    "GRID2OP_MINIMUM_EVALUATION_UNITS",
    "GRID2OP_MINIMUM_PROSPECTIVE_VALIDATION_UNITS",
    "GRID2OP_NATIVE_ACTION_IDS",
    "GRID2OP_RECEIVER_GAUGE_IDS",
    "GRID2OP_SIMULTANEOUS_FAMILY_IDS",
    "GRID2OP_SOURCE_SUPPORTED_ACTION_KIND_IDS",
    "build_grid2op_target_design",
    "freeze_grid2op_power",
    "grid2op_denominator_candidates",
    "grid2op_mapping_candidates",
]
