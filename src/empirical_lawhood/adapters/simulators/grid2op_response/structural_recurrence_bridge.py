'Grid2Op-native lowering into the unchanged frozen margin structural recurrence forecast contracts.\n\nThe simulator runtime never imports this module.  It emits only target-native\nepisode records.  This evaluator-side bridge consumes compact complete-chronic\npanels after the applicable access barrier and preserves the frozen margin structural recurrence forecast\npredictor unchanged.\n'

from __future__ import annotations

from decimal import Decimal
from statistics import median

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateDenominatorCandidate, IndependentSubstrateTargetMappingCandidate, IndependentSubstrateTargetKind
from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceActionOutcome, StructuralRecurrenceNativeThreshold, StructuralRecurrenceQualificationBranch, StructuralRecurrenceSplitRoster, StructuralRecurrenceSourceQualification, StructuralRecurrenceStageEvidence, StructuralRecurrenceTargetDesignFreeze, StructuralRecurrenceTargetStage, StructuralRecurrenceUnitObservation
from empirical_lawhood.adapters.methods.structural_recurrence import UNIVERSAL_ROLES, NativeRoleBinding, NativeRoleCompatibilityRecord, TargetLevel, TargetSlotRecord, UniversalRole, structural_ontology
from empirical_lawhood.adapters.methods.structured_target import IndependentSubstrateActionDeliveryStatus, IndependentSubstrateCompleteTargetPanel, IndependentSubstrateCompleteTargetUnit, IndependentSubstrateStructuredPowerFreeze, IndependentSubstrateTargetPhase, IndependentSubstrateUnitStatus
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import require_sorted_unique_ids

from .contracts import Grid2OpSourceBinding
from .source import Grid2OpSourceQualificationDisposition, Grid2OpSourceQualification


GRID2OP_REQUIRED_STRUCTURAL_RECURRENCE_THRESHOLD_IDS = (
    "denominator-common-max",
    "denominator-view-local-max",
    "direct-composed-max",
    "effort-max",
    "history-close-max",
    "realization-error-max",
    "sink-margin-min",
    "target-effect-min",
    "timing-offset-max",
    "unit-pass-rate-min",
)

GRID2OP_STRUCTURAL_RECURRENCE_THRESHOLDS = (
    StructuralRecurrenceNativeThreshold(
        threshold_id="denominator-common-max",
        native_metric_id="denominator-common-max",
        value=Decimal("0.05"),
        native_unit="maximum-rho",
        direction="AT_MOST",
    ),
    StructuralRecurrenceNativeThreshold(
        threshold_id="denominator-view-local-max",
        native_metric_id="denominator-view-local-max",
        value=Decimal("0.15"),
        native_unit="maximum-rho",
        direction="AT_MOST",
    ),
    StructuralRecurrenceNativeThreshold(
        threshold_id="direct-composed-max",
        native_metric_id="direct-composed-max",
        value=Decimal("0.01"),
        native_unit="maximum-rho",
        direction="AT_MOST",
    ),
    StructuralRecurrenceNativeThreshold(
        threshold_id="effort-max",
        native_metric_id="effort-max",
        value=Decimal(2),
        native_unit="disconnected-line-count",
        direction="AT_MOST",
    ),
    StructuralRecurrenceNativeThreshold(
        threshold_id="history-close-max",
        native_metric_id="history-close-max",
        value=Decimal("0.02"),
        native_unit="maximum-rho",
        direction="AT_MOST",
    ),
    StructuralRecurrenceNativeThreshold(
        threshold_id="realization-error-max",
        native_metric_id="realization-error-max",
        value=Decimal(0),
        native_unit="categorical-code",
        direction="AT_MOST",
    ),
    StructuralRecurrenceNativeThreshold(
        threshold_id="sink-margin-min",
        native_metric_id="sink-margin-min",
        value=Decimal(0),
        native_unit="maximum-rho",
        direction="AT_LEAST",
    ),
    StructuralRecurrenceNativeThreshold(
        threshold_id="target-effect-min",
        native_metric_id="target-effect-min",
        value=Decimal("0.01"),
        native_unit="maximum-rho-relief",
        direction="AT_LEAST",
    ),
    StructuralRecurrenceNativeThreshold(
        threshold_id="timing-offset-max",
        native_metric_id="timing-offset-max",
        value=Decimal(0),
        native_unit="native-step",
        direction="AT_MOST",
    ),
    StructuralRecurrenceNativeThreshold(
        threshold_id="unit-pass-rate-min",
        native_metric_id="unit-pass-rate-min",
        value=Decimal("0.50"),
        native_unit="proportion",
        direction="AT_LEAST",
    ),
)

_ROLE_UNITS = {
    UniversalRole.ACTION_REALIZATION: "categorical-code",
    UniversalRole.AUTHORITY: "boolean",
    UniversalRole.DYNAMICS: "native-step",
    UniversalRole.EFFORT: "disconnected-line-count",
    UniversalRole.HOLD: "categorical-code",
    UniversalRole.HORIZON_CLOCK: "native-step",
    UniversalRole.NATIVE_ACTION: "categorical-code",
    UniversalRole.PREPARED_DENOMINATOR: "held-source-fingerprint",
    UniversalRole.PRESERVATION: "maximum-rho",
    UniversalRole.REACHABILITY: "boolean",
    UniversalRole.RECEIVER_SINK: "maximum-rho",
    UniversalRole.RECEIVER_TARGET: "maximum-rho-relief",
    UniversalRole.RETAINED_HISTORY: "native-chronic-state",
    UniversalRole.SUPPORT: "boolean",
    UniversalRole.UNCERTAINTY: "complete-chronic",
    UniversalRole.VALIDITY: "boolean",
}


def _native_compatibility(target_slot: TargetSlotRecord) -> NativeRoleCompatibilityRecord:
    bindings = tuple(
        NativeRoleBinding(
            binding_id=f"binding.grid2op.structural-recurrence.{role.value.lower().replace('_', '-')}",
            native_object_id=f"native.grid2op.{role.value.lower().replace('_', '-')}",
            role=role,
            native_unit=_ROLE_UNITS[role],
            native_direction="TARGET_NATIVE_SIGNED_OR_CATEGORICAL",
            clock_id=f"clock.grid2op.{role.value.lower().replace('_', '-')}",
            missingness_rule="Missing native role remains typed and lowers the ceiling.",
            claim_ceiling=EvidenceCeiling.ADMISSION,
            available=True,
            reuse_compatibility_id=None,
        )
        for role in UNIVERSAL_ROLES
    )
    return NativeRoleCompatibilityRecord(
        map_id='compatibility.grid2op.margin-structural-recurrence-forecast',
        target_slot_id=target_slot.target_slot_id,
        ontology=target_slot.ontology,
        bindings=bindings,
        required_new_role_ids=(),
        map_valid=True,
        reason_codes=(),
    )


def build_grid2op_structural_recurrence_source_qualification(
    *,
    source: Grid2OpSourceBinding,
    qualification: Grid2OpSourceQualification,
) -> StructuralRecurrenceSourceQualification:
    """Project the excluded source probe without counting it as target evidence."""

    if (
        qualification.disposition is not Grid2OpSourceQualificationDisposition.PASS
        or qualification.target_source != ObjectIdentity.from_record(source.binding_id, source)
    ):
        raise ValueError('Grid2Op structural recurrence source qualification did not pass for this source')
    return StructuralRecurrenceSourceQualification(
        qualification_id='qualification.grid2op.margin-structural-recurrence-forecast-source',
        target_slot_id="target-slot.independent-substrate-grounding.grid2op",
        primary_source_id=source.binding_id,
        primary_source_available=True,
        primary_reason_codes=(),
        selected_source_id=source.binding_id,
        selected_source_fingerprint=source.fingerprint(),
        evidence_world_id="resettable-simulator.grid2op",
        branch=StructuralRecurrenceQualificationBranch.ENTER_LAW_QUALIFICATION,
        local_execution=True,
        network_used=False,
        dependency_mutated=False,
        requested_action_observable=True,
        accepted_action_observable=True,
        applied_action_observable=True,
        realized_action_observable=True,
        receiver_available=True,
        exact_reset_available=True,
        scientific_outcome_accessed=False,
    )


def build_grid2op_structural_recurrence_target_design(
    *,
    source: Grid2OpSourceBinding,
    qualification: StructuralRecurrenceSourceQualification,
    mapping_candidate: IndependentSubstrateTargetMappingCandidate,
    denominator_candidate: IndependentSubstrateDenominatorCandidate,
    power_freeze: IndependentSubstrateStructuredPowerFreeze,
    selected_source_implementation: ObjectIdentity,
    evaluator_implementation: ObjectIdentity,
) -> StructuralRecurrenceTargetDesignFreeze:
    """Freeze one Grid2Op member of the predeclared outcome-blind scientific grammar design roster."""

    if mapping_candidate.target_slot is not IndependentSubstrateTargetKind.GRID2OP:
        raise ValueError('Grid2Op structural recurrence design received another target mapping')
    if denominator_candidate.impossible:
        raise ValueError("impossible Grid2Op denominator cannot issue")
    if not power_freeze.joint_precision_passed:
        raise ValueError('Grid2Op structural recurrence design requires joint precision')
    if tuple(value.threshold_id for value in GRID2OP_STRUCTURAL_RECURRENCE_THRESHOLDS) != (
        GRID2OP_REQUIRED_STRUCTURAL_RECURRENCE_THRESHOLD_IDS
    ):
        raise ValueError('Grid2Op structural recurrence threshold roster differs')
    target_slot = TargetSlotRecord(
        target_slot_id="target-slot.independent-substrate-grounding.grid2op",
        target_class_id="target-class.grid2op-independent-system",
        canonical_order=1,
        target_level=TargetLevel.ADMISSION,
        evidence_world_id="resettable-simulator.grid2op",
        source=ObjectIdentity.from_record(source.binding_id, source),
        ontology=ObjectIdentity.from_record(
            'ontology.categorical-structural-recurrence',
            structural_ontology(),
        ),
        reserve_source_ids=(),
        outcome_naive_at_freeze=True,
        evaluation_outcomes_accessed=False,
    )
    rosters = tuple(
        StructuralRecurrenceSplitRoster(
            roster_id=f"roster.grid2op.margin-structural-recurrence-forecast.{stage.value.lower()}",
            stage=stage,
            unit_ids=unit_ids,
            seeds=tuple(seed_offset + index for index in range(len(unit_ids))),
        )
        for stage, unit_ids, seed_offset in (
            (StructuralRecurrenceTargetStage.DEVELOPMENT, power_freeze.development_unit_ids, 10_000),
            (StructuralRecurrenceTargetStage.EVALUATION, power_freeze.evaluation_unit_ids, 20_000),
            (StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION, power_freeze.prospective_validation_unit_ids, 30_000),
        )
    )
    mapping = {value.structural_recurrence_role_id: value for value in mapping_candidate.role_bindings}
    return StructuralRecurrenceTargetDesignFreeze(
        design_id='design.independent-substrate-grounding.grid2op.structural-recurrence',
        target_slot=target_slot,
        qualification=qualification,
        prepared_denominator=(
            f"{denominator_candidate.candidate_id}:"
            f"{','.join(denominator_candidate.factor_ids)}:"
            f"{','.join(denominator_candidate.stratum_ids)}"
        ),
        retained_history=",".join(mapping["H"].target_native_role_ids),
        horizon_clock=",".join(mapping["tau"].target_native_role_ids),
        independent_unit="one exact held native chronic",
        observation_operator=",".join(mapping["R"].target_native_role_ids),
        native_action_ids=(
            "disconnect-line-000",
            "disconnect-line-001",
            "disconnect-lines-000-001",
            "hold",
        ),
        receiver_ids=("maximum-rho", "topology-state"),
        thresholds=GRID2OP_STRUCTURAL_RECURRENCE_THRESHOLDS,
        rosters=rosters,
        compatibility=_native_compatibility(target_slot),
        selected_source_implementation=selected_source_implementation,
        evaluator_implementation=evaluator_implementation,
        maximum_level=TargetLevel.ADMISSION,
        frozen_before_development=True,
        protected_outcome_access_count=0,
    )


def _points(
    unit: IndependentSubstrateCompleteTargetUnit,
    *,
    action_id: str,
    receiver_id: str,
) -> dict[int, Decimal]:
    branch_id = f"branch.grid2op.{action_id}"
    return {
        value.horizon_tick: value.value
        for value in unit.receiver_points
        if f".{branch_id}." in value.point_id
        and value.receiver_id == receiver_id
        and value.valid
        and value.value is not None
    }


def _ledger(unit: IndependentSubstrateCompleteTargetUnit, action_id: str):
    return next(value for value in unit.action_ledgers if value.native_action_id == action_id)


def _effect(unit: IndependentSubstrateCompleteTargetUnit, action_id: str, horizon: int) -> Decimal | None:
    hold = _points(unit, action_id="hold", receiver_id="maximum-rho").get(horizon)
    acted = _points(unit, action_id=action_id, receiver_id="maximum-rho").get(horizon)
    if hold is None or acted is None:
        return None
    return hold - acted


def _panel_denominator_distance(
    panel: IndependentSubstrateCompleteTargetPanel,
    action_ids: tuple[str, ...],
) -> Decimal:
    deviations = []
    for action_id in action_ids:
        values = tuple(
            value for unit in panel.units if (value := _effect(unit, action_id, 6)) is not None
        )
        if not values:
            return Decimal(1)
        center = median(values)
        deviations.extend(abs(value - center) for value in values)
    return median(deviations) if deviations else Decimal(0)


def lower_grid2op_panel_to_structural_recurrence(
    *,
    design: StructuralRecurrenceTargetDesignFreeze,
    power_freeze: IndependentSubstrateStructuredPowerFreeze,
    panel: IndependentSubstrateCompleteTargetPanel,
    execution_authority_verified: bool,
    nonhold_action_ids: tuple[str, ...] | None = None,
) -> StructuralRecurrenceStageEvidence:
    """Lower every issued chronic, including sinks and observation failures."""

    stage = {
        IndependentSubstrateTargetPhase.DEVELOPMENT: StructuralRecurrenceTargetStage.DEVELOPMENT,
        IndependentSubstrateTargetPhase.EVALUATION: StructuralRecurrenceTargetStage.EVALUATION,
        IndependentSubstrateTargetPhase.PROSPECTIVE_VALIDATION: StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION,
    }[panel.phase]
    roster = design.roster(stage)
    if (
        panel.planned_unit_ids != roster.unit_ids
        or tuple(value.unit_id for value in panel.units) != roster.unit_ids
    ):
        raise ValueError('Grid2Op structural recurrence panel differs from its issued chronic roster')
    chart = tuple(value for value in design.native_action_ids if value != "hold")
    nonhold = chart if nonhold_action_ids is None else nonhold_action_ids
    if tuple(sorted(set(nonhold))) != nonhold or not set(nonhold) <= set(chart):
        raise ValueError("Grid2Op prospective validation action subset differs from the frozen chart")
    action_ids = tuple(sorted((*nonhold, "hold")))
    denominator_distance = _panel_denominator_distance(panel, nonhold)
    seed_by_unit = dict(zip(roster.unit_ids, roster.seeds, strict=True))
    observations = []
    for unit in panel.units:
        final_effects = {action_id: _effect(unit, action_id, 6) for action_id in nonhold}
        atomic_sum = sum(
            (
                final_effects.get("disconnect-line-000") or Decimal(0),
                final_effects.get("disconnect-line-001") or Decimal(0),
            ),
            Decimal(0),
        )
        joint = final_effects.get("disconnect-lines-000-001")
        composition = abs(joint - atomic_sum) if joint is not None else Decimal(1)
        outcomes = []
        for action_id in action_ids:
            ledger = _ledger(unit, action_id)
            rho = _points(unit, action_id=action_id, receiver_id="maximum-rho")
            effect_1 = Decimal(0) if action_id == "hold" else _effect(unit, action_id, 1)
            effect_3 = Decimal(0) if action_id == "hold" else _effect(unit, action_id, 3)
            effect_6 = Decimal(0) if action_id == "hold" else _effect(unit, action_id, 6)
            complete = unit.status is IndependentSubstrateUnitStatus.COMPLETE and all(
                value is not None for value in (effect_1, effect_3, effect_6)
            )
            realized = ledger.delivery_status is IndependentSubstrateActionDeliveryStatus.REALIZED
            sink_margin = min((Decimal(1) - value for value in rho.values()), default=Decimal(-1))
            target_effect = effect_6 if effect_6 is not None else Decimal(0)
            target_supported = action_id == "hold" or target_effect >= design.threshold(
                "target-effect-min"
            )
            preservation = complete and sink_margin >= 0
            reasons = set(unit.reason_codes) | set(ledger.reason_codes)
            if not complete:
                reasons.add("GRID2OP_COMPLETE_CHRONIC_OBSERVATION_UNAVAILABLE")
            if not target_supported:
                reasons.add("GRID2OP_TARGET_EFFECT_BELOW_FROZEN_MARGIN")
            if not preservation:
                reasons.add("GRID2OP_THERMAL_SINK_OR_PRESERVATION_FAILURE")
            if not execution_authority_verified:
                reasons.add("EXECUTION_AUTHORITY_UNVERIFIED")
            accepted = Decimal(ledger.accepted_action_code is not None)
            applied = Decimal(ledger.applied_action_code is not None)
            realized_value = Decimal(realized)
            outcomes.append(
                StructuralRecurrenceActionOutcome(
                    outcome_id=f"{unit.unit_id}.{action_id}",
                    action_id=action_id,
                    requested_action=Decimal(1),
                    accepted_action=accepted,
                    applied_action=applied,
                    realized_action=realized_value,
                    target_effect=target_effect,
                    sink_margin=sink_margin,
                    effort=Decimal(
                        {
                            "disconnect-line-000": 1,
                            "disconnect-line-001": 1,
                            "disconnect-lines-000-001": 2,
                            "hold": 0,
                        }[action_id]
                    ),
                    realization_error=max(
                        abs(Decimal(1) - accepted),
                        abs(accepted - applied),
                        abs(applied - realized_value),
                    ),
                    denominator_distance=denominator_distance,
                    history_checkpoint_distance=(
                        abs(effect_1 - effect_3)
                        if effect_1 is not None and effect_3 is not None
                        else Decimal(1)
                    ),
                    history_future_distance=(
                        abs(effect_3 - effect_6)
                        if effect_3 is not None and effect_6 is not None
                        else Decimal(1)
                    ),
                    direct_composed_distance=(Decimal(0) if action_id == "hold" else composition),
                    timing_offset=Decimal(0) if ledger.clock_complete else Decimal(1),
                    support_preserved=complete and target_supported,
                    validity_passed=complete,
                    preservation_passed=preservation,
                    dynamics_passed=complete and not unit.sink_codes,
                    reachability_passed=realized,
                    authority_passed=execution_authority_verified,
                    uncertainty_evaluable=(
                        complete
                        and power_freeze.joint_precision_passed
                        and not power_freeze.panel_envelope_limited
                    ),
                    reason_codes=tuple(sorted(reasons)),
                )
            )
        observations.append(
            StructuralRecurrenceUnitObservation(
                unit_id=unit.unit_id,
                target_slot_id=design.target_slot.target_slot_id,
                stage=stage,
                seed=seed_by_unit[unit.unit_id],
                preparation_fingerprint=unit.fingerprint(),
                outcomes=tuple(sorted(outcomes, key=lambda value: value.outcome_id)),
            )
        )
    require_sorted_unique_ids(observations, attribute="unit_id", field_name="observations")
    ordered = tuple(sorted(observations, key=lambda value: value.unit_id))
    return StructuralRecurrenceStageEvidence(
        evidence_id=f"evidence.grid2op.margin-structural-recurrence-forecast.{stage.value.lower()}",
        target_design=ObjectIdentity.from_record(design.design_id, design),
        target_slot_id=design.target_slot.target_slot_id,
        stage=stage,
        units=ordered,
        independent_unit_count=len(ordered),
        nested_action_outcome_count=sum(len(value.outcomes) for value in ordered),
        source_implementation=design.selected_source_implementation,
        outcome_access=(
            OutcomeAccess.DEVELOPMENT_VISIBLE
            if stage is StructuralRecurrenceTargetStage.DEVELOPMENT
            else OutcomeAccess.EVALUATION_SEALED
        ),
    )


__all__ = [
    'GRID2OP_STRUCTURAL_RECURRENCE_THRESHOLDS',
    'GRID2OP_REQUIRED_STRUCTURAL_RECURRENCE_THRESHOLD_IDS',
    'build_grid2op_structural_recurrence_source_qualification',
    'build_grid2op_structural_recurrence_target_design',
    'lower_grid2op_panel_to_structural_recurrence',
]
