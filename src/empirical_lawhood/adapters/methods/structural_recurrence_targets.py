'Typed lifecycle and target-independent reduction for fresh categorical structural recurrence targets.\n\nThis module never executes a simulator and never opens an artifact path.  Target\nadapters emit the native, unit-level records defined here.  Development lowers\nthose records to the frozen structural recurrence ontology; the separately implemented target\nevaluator lowers sealed evaluation/prospective validation evidence to an observed signature.\n'

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from statistics import median
from typing import ClassVar

from empirical_lawhood.adapters.methods import structural_recurrence as core
from empirical_lawhood.kernel.evidence import EvidenceRung, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)


class StructuralRecurrenceTargetStage(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"
    PROSPECTIVE_VALIDATION = "PROSPECTIVE_VALIDATION"


class StructuralRecurrenceQualificationBranch(StrEnum):
    ENTER_LAW_QUALIFICATION = "ENTER_LAW_QUALIFICATION"
    MEASUREMENT_PREDICTED_STOP = "MEASUREMENT_PREDICTED_STOP"
    TARGET_CLASS_SOURCE_UNAVAILABLE = "TARGET_CLASS_SOURCE_UNAVAILABLE"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    RESOURCE_STOP = "RESOURCE_STOP"


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceSourceQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-recurrence-source-qualification'

    qualification_id: str
    target_slot_id: str
    primary_source_id: str
    primary_source_available: bool
    primary_reason_codes: tuple[str, ...]
    selected_source_id: str
    selected_source_fingerprint: str
    evidence_world_id: str
    branch: StructuralRecurrenceQualificationBranch
    local_execution: bool
    network_used: bool
    dependency_mutated: bool
    requested_action_observable: bool
    accepted_action_observable: bool
    applied_action_observable: bool
    realized_action_observable: bool
    receiver_available: bool
    exact_reset_available: bool
    scientific_outcome_accessed: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("qualification_id", self.qualification_id),
            ("target_slot_id", self.target_slot_id),
            ("primary_source_id", self.primary_source_id),
            ("selected_source_id", self.selected_source_id),
            ("evidence_world_id", self.evidence_world_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.primary_reason_codes, field_name="primary_reason_codes")
        validate_sha256(self.selected_source_fingerprint, field_name="selected_source_fingerprint")
        if self.network_used or self.dependency_mutated or self.scientific_outcome_accessed:
            raise ValueError("source qualification crossed the outcome-blind local gate")
        if self.branch is StructuralRecurrenceQualificationBranch.ENTER_LAW_QUALIFICATION:
            if not all(
                (
                    self.local_execution,
                    self.requested_action_observable,
                    self.accepted_action_observable,
                    self.applied_action_observable,
                    self.realized_action_observable,
                    self.receiver_available,
                    self.exact_reset_available,
                )
            ):
                raise ValueError("ENTER_LAW_QUALIFICATION requires the complete local source/action surface")


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceNativeThreshold(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-recurrence-native-threshold'

    threshold_id: str
    native_metric_id: str
    value: Decimal
    native_unit: str
    direction: str

    def __post_init__(self) -> None:
        validate_stable_id(self.threshold_id, field_name="threshold_id")
        validate_stable_id(self.native_metric_id, field_name="native_metric_id")
        validate_decimal(self.value, field_name="value")
        validate_nonempty(self.native_unit, field_name="native_unit")
        if self.direction not in {"AT_LEAST", "AT_MOST", "ABS_AT_MOST"}:
            raise ValueError("unknown native threshold direction")


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceSplitRoster(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-recurrence-split-roster'

    roster_id: str
    stage: StructuralRecurrenceTargetStage
    unit_ids: tuple[str, ...]
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        require_sorted_unique_strings(self.unit_ids, field_name="unit_ids", allow_empty=False)
        if len(self.unit_ids) != len(self.seeds) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("split roster requires one unique seed per independent unit")
        if any(value < 0 for value in self.seeds):
            raise ValueError("seeds must be nonnegative")


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceTargetDesignFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-recurrence-target-design-freeze'

    design_id: str
    target_slot: core.TargetSlotRecord
    qualification: StructuralRecurrenceSourceQualification
    prepared_denominator: str
    retained_history: str
    horizon_clock: str
    independent_unit: str
    observation_operator: str
    native_action_ids: tuple[str, ...]
    receiver_ids: tuple[str, ...]
    thresholds: tuple[StructuralRecurrenceNativeThreshold, ...]
    rosters: tuple[StructuralRecurrenceSplitRoster, ...]
    compatibility: core.NativeRoleCompatibilityRecord
    selected_source_implementation: ObjectIdentity
    evaluator_implementation: ObjectIdentity
    maximum_level: core.TargetLevel
    frozen_before_development: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.design_id, field_name="design_id")
        for name in (
            "prepared_denominator",
            "retained_history",
            "horizon_clock",
            "independent_unit",
            "observation_operator",
        ):
            validate_nonempty(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.native_action_ids, field_name="native_action_ids", allow_empty=False)
        require_sorted_unique_strings(self.receiver_ids, field_name="receiver_ids", allow_empty=False)
        require_sorted_unique_ids(self.thresholds, attribute="threshold_id", field_name="thresholds")
        require_sorted_unique_ids(self.rosters, attribute="roster_id", field_name="rosters")
        if {value.stage for value in self.rosters} != set(StructuralRecurrenceTargetStage):
            raise ValueError('target design requires development, evaluation and prospective validation rosters')
        all_unit_ids = tuple(unit_id for roster in self.rosters for unit_id in roster.unit_ids)
        all_seeds = tuple(seed for roster in self.rosters for seed in roster.seeds)
        if len(set(all_unit_ids)) != len(all_unit_ids) or len(set(all_seeds)) != len(all_seeds):
            raise ValueError("target split rosters must be disjoint in unit and seed identity")
        if "hold" not in self.native_action_ids:
            raise ValueError("target action chart requires exact hold")
        if self.qualification.target_slot_id != self.target_slot.target_slot_id:
            raise ValueError("qualification and target slot differ")
        if self.compatibility.target_slot_id != self.target_slot.target_slot_id:
            raise ValueError("compatibility map and target slot differ")
        if not self.frozen_before_development or self.protected_outcome_access_count:
            raise ValueError("target design must freeze before development and protected outcomes")

    def roster(self, stage: StructuralRecurrenceTargetStage) -> StructuralRecurrenceSplitRoster:
        return next(value for value in self.rosters if value.stage is stage)

    def threshold(self, metric_id: str) -> Decimal:
        return next(value.value for value in self.thresholds if value.native_metric_id == metric_id)


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceActionOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-recurrence-action-outcome'

    outcome_id: str
    action_id: str
    requested_action: Decimal
    accepted_action: Decimal
    applied_action: Decimal
    realized_action: Decimal
    target_effect: Decimal
    sink_margin: Decimal
    effort: Decimal
    realization_error: Decimal
    denominator_distance: Decimal
    history_checkpoint_distance: Decimal
    history_future_distance: Decimal
    direct_composed_distance: Decimal
    timing_offset: Decimal
    support_preserved: bool
    validity_passed: bool
    preservation_passed: bool
    dynamics_passed: bool
    reachability_passed: bool
    authority_passed: bool
    uncertainty_evaluable: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.outcome_id, field_name="outcome_id")
        validate_stable_id(self.action_id, field_name="action_id")
        for name in (
            "requested_action",
            "accepted_action",
            "applied_action",
            "realized_action",
            "target_effect",
            "sink_margin",
            "effort",
            "realization_error",
            "denominator_distance",
            "history_checkpoint_distance",
            "history_future_distance",
            "direct_composed_distance",
            "timing_offset",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if min(
            self.realization_error,
            self.denominator_distance,
            self.history_checkpoint_distance,
            self.history_future_distance,
            self.direct_composed_distance,
            self.timing_offset,
            self.effort,
        ) < 0:
            raise ValueError("distance, timing, error and effort fields must be nonnegative")


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceUnitObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-recurrence-unit-observation'

    unit_id: str
    target_slot_id: str
    stage: StructuralRecurrenceTargetStage
    seed: int
    preparation_fingerprint: str
    outcomes: tuple[StructuralRecurrenceActionOutcome, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.unit_id, field_name="unit_id")
        validate_stable_id(self.target_slot_id, field_name="target_slot_id")
        validate_sha256(self.preparation_fingerprint, field_name="preparation_fingerprint")
        require_sorted_unique_ids(self.outcomes, attribute="outcome_id", field_name="outcomes")
        if self.seed < 0 or not self.outcomes:
            raise ValueError("unit requires a nonnegative seed and native outcomes")


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceStageEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-recurrence-stage-evidence'

    evidence_id: str
    target_design: ObjectIdentity
    target_slot_id: str
    stage: StructuralRecurrenceTargetStage
    units: tuple[StructuralRecurrenceUnitObservation, ...]
    independent_unit_count: int
    nested_action_outcome_count: int
    source_implementation: ObjectIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        validate_stable_id(self.target_slot_id, field_name="target_slot_id")
        require_sorted_unique_ids(self.units, attribute="unit_id", field_name="units")
        if any(value.target_slot_id != self.target_slot_id or value.stage is not self.stage for value in self.units):
            raise ValueError("stage evidence crosses target or split")
        if self.independent_unit_count != len(self.units):
            raise ValueError("independent unit count is not unit-derived")
        if self.nested_action_outcome_count != sum(len(value.outcomes) for value in self.units):
            raise ValueError("nested action count is not unit-derived")
        expected = {
            StructuralRecurrenceTargetStage.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            StructuralRecurrenceTargetStage.EVALUATION: OutcomeAccess.EVALUATION_SEALED,
            StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION: OutcomeAccess.EVALUATION_SEALED,
        }[self.stage]
        if self.outcome_access is not expected:
            raise ValueError("stage evidence has the wrong outcome-access boundary")


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceDonorCompilerFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-recurrence-donor-compiler-freeze'

    freeze_id: str
    design_freeze: ObjectIdentity
    conformance: ObjectIdentity
    donor_corpus: core.DonorCorpusIdentity
    controlling_donor_ids: tuple[str, ...]
    leave_one_donor_rule_fingerprints: tuple[str, ...]
    compiler_source: ObjectIdentity
    decision_table_sha256: str
    donor_payload_in_target_config: bool
    frozen: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        require_sorted_unique_strings(self.controlling_donor_ids, field_name="controlling_donor_ids", allow_empty=False)
        require_sorted_unique_strings(
            self.leave_one_donor_rule_fingerprints,
            field_name="leave_one_donor_rule_fingerprints",
            allow_empty=False,
        )
        validate_sha256(self.decision_table_sha256, field_name="decision_table_sha256")
        if len(self.leave_one_donor_rule_fingerprints) != len(self.controlling_donor_ids):
            raise ValueError("leave-one-donor sensitivity must cover every donor")
        if self.donor_payload_in_target_config or not self.frozen:
            raise ValueError("donor compiler freeze is not target-independent")


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceDevelopmentHandoff(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-recurrence-development-handoff'

    handoff_id: str
    target_design: ObjectIdentity
    development_evidence: ObjectIdentity
    prediction_input: core.StructuralPredictionInput
    selected_action_id: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.handoff_id, field_name="handoff_id")
        validate_stable_id(self.selected_action_id, field_name="selected_action_id")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("development handoff must remain development-visible")


@dataclass(frozen=True, slots=True)
class StructuralRecurrencePredictionIssue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-recurrence-prediction-issue'

    issue_id: str
    donor_compiler_freeze: ObjectIdentity
    target_design: ObjectIdentity
    development_handoff: ObjectIdentity
    prediction: core.StructuralPrediction
    comparators: core.PredictiveComparatorPanel
    evaluator_implementation: ObjectIdentity
    target_specific_falsifier_ids: tuple[str, ...]
    published_before_evaluation: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.issue_id, field_name="issue_id")
        require_sorted_unique_strings(
            self.target_specific_falsifier_ids,
            field_name="target_specific_falsifier_ids",
            allow_empty=False,
        )
        if not self.published_before_evaluation or self.protected_outcome_access_count:
            raise ValueError("prediction issue is not prospective")
        if self.prediction.status is not core.PredictionStatus.ISSUED:
            raise ValueError("nonrestrictive prediction cannot be issued")
        if self.prediction.prediction_input != self.comparators.prediction_input:
            raise ValueError("prediction and comparator inputs differ")


def _proportion(values: tuple[bool, ...]) -> Decimal:
    return Decimal(sum(values)) / Decimal(len(values))


def _median(values: tuple[Decimal, ...]) -> Decimal:
    return median(values)


def _all_outcomes(evidence: StructuralRecurrenceStageEvidence, action_id: str) -> tuple[StructuralRecurrenceActionOutcome, ...]:
    return tuple(
        next(outcome for outcome in unit.outcomes if outcome.action_id == action_id)
        for unit in evidence.units
    )


def _categorical_structure(
    evidence: StructuralRecurrenceStageEvidence,
    design: StructuralRecurrenceTargetDesignFreeze,
) -> tuple[core.DenominatorStructure, core.HistoryClockQuotient, core.SupportTransport]:
    values = tuple(outcome for unit in evidence.units for outcome in unit.outcomes)
    denom = _median(tuple(value.denominator_distance for value in values))
    common_max = design.threshold("denominator-common-max")
    view_max = design.threshold("denominator-view-local-max")
    denominator = (
        core.DenominatorStructure.COMMON_LAW
        if denom <= common_max
        else core.DenominatorStructure.VIEW_LOCAL
        if denom <= view_max
        else core.DenominatorStructure.INCOMPATIBLE
    )
    close_max = design.threshold("history-close-max")
    checkpoint_close = _proportion(
        tuple(value.history_checkpoint_distance <= close_max for value in values)
    )
    future_close = _proportion(
        tuple(value.history_future_distance <= close_max for value in values)
    )
    timing_bad = _proportion(
        tuple(value.timing_offset > design.threshold("timing-offset-max") for value in values)
    )
    pass_rate = design.threshold("unit-pass-rate-min")
    if timing_bad > Decimal(1) - pass_rate:
        history = core.HistoryClockQuotient.TIMING_CONFOUNDED
    elif checkpoint_close >= pass_rate and future_close >= pass_rate:
        history = core.HistoryClockQuotient.CLOSE_CLOSED
    elif checkpoint_close >= pass_rate and future_close < pass_rate:
        history = core.HistoryClockQuotient.CLOSE_DIVERGED
    else:
        history = core.HistoryClockQuotient.MIXED
    compose_max = design.threshold("direct-composed-max")
    compose_pass = _proportion(tuple(value.direct_composed_distance <= compose_max for value in values))
    support_pass = _proportion(tuple(value.support_preserved for value in values))
    if compose_pass < pass_rate:
        support = core.SupportTransport.OPPOSED
    elif support_pass == Decimal(1):
        support = core.SupportTransport.EQUALITY
    elif support_pass >= pass_rate:
        support = core.SupportTransport.CONSERVATIVE_INCLUSION
    elif support_pass > 0:
        support = core.SupportTransport.BOUNDARY_LOSS
    else:
        support = core.SupportTransport.OPPOSED
    return denominator, history, support


def _role_status(
    role: core.UniversalRole,
    outcomes: tuple[StructuralRecurrenceActionOutcome, ...],
    design: StructuralRecurrenceTargetDesignFreeze,
) -> core.OperandStatus:
    pass_rate = design.threshold("unit-pass-rate-min")
    if role is core.UniversalRole.SUPPORT:
        passed = _proportion(tuple(value.support_preserved for value in outcomes)) >= pass_rate
    elif role is core.UniversalRole.RECEIVER_TARGET:
        minimum = design.threshold("target-effect-min")
        passed = _median(tuple(value.target_effect for value in outcomes)) >= minimum
    elif role is core.UniversalRole.RECEIVER_SINK:
        minimum = design.threshold("sink-margin-min")
        passed = _proportion(tuple(value.sink_margin >= minimum for value in outcomes)) >= pass_rate
    elif role is core.UniversalRole.EFFORT:
        passed = _proportion(tuple(value.effort <= design.threshold("effort-max") for value in outcomes)) >= pass_rate
    elif role is core.UniversalRole.VALIDITY:
        passed = _proportion(tuple(value.validity_passed for value in outcomes)) >= pass_rate
    elif role is core.UniversalRole.UNCERTAINTY:
        passed = _proportion(tuple(value.uncertainty_evaluable for value in outcomes)) >= pass_rate
    elif role is core.UniversalRole.PRESERVATION:
        passed = _proportion(tuple(value.preservation_passed for value in outcomes)) >= pass_rate
    elif role is core.UniversalRole.DYNAMICS:
        passed = _proportion(tuple(value.dynamics_passed for value in outcomes)) >= pass_rate
    elif role is core.UniversalRole.REACHABILITY:
        passed = _proportion(tuple(value.reachability_passed for value in outcomes)) >= pass_rate
    elif role is core.UniversalRole.AUTHORITY:
        passed = _proportion(tuple(value.authority_passed for value in outcomes)) == Decimal(1)
    else:  # pragma: no cover - closed by ADMISSION_ROLES
        raise AssertionError(role)
    return core.OperandStatus.PASS if passed else core.OperandStatus.FAIL


def build_prediction_input(
    *,
    design: StructuralRecurrenceTargetDesignFreeze,
    evidence: StructuralRecurrenceStageEvidence,
    donor_corpus: core.DonorCorpusIdentity,
) -> tuple[core.StructuralPredictionInput, str]:
    """Lower development-visible native evidence without target thresholds/rows leaking."""

    if evidence.stage is not StructuralRecurrenceTargetStage.DEVELOPMENT:
        raise ValueError("prediction input requires development evidence")
    denominator, history, support = _categorical_structure(evidence, design)
    facts: list[core.ActionCandidateFact] = []
    ranked = []
    for action_id in design.native_action_ids:
        if action_id == "hold":
            continue
        outcomes = _all_outcomes(evidence, action_id)
        ranked.append((action_id, _median(tuple(value.target_effect for value in outcomes))))
    ranks = {
        action_id: index + 1
        for index, (action_id, _) in enumerate(sorted(ranked, key=lambda item: (-item[1], item[0])))
    }
    evidence_identity = ObjectIdentity.from_record(evidence.evidence_id, evidence)
    for action_id, effect in sorted(ranked):
        outcomes = _all_outcomes(evidence, action_id)
        realization = (
            core.ActionRole.REALIZATION_QUALIFIED
            if _proportion(
                tuple(value.realization_error <= design.threshold("realization-error-max") for value in outcomes)
            ) >= design.threshold("unit-pass-rate-min")
            else core.ActionRole.MISMATCH
        )
        operands = tuple(
            core.AdmissionOperandFact(
                operand_id=f"{design.target_slot.target_slot_id}.dev.{action_id}.{index:02d}",
                role=role,
                status=(status := _role_status(role, outcomes, design)),
                evidence=evidence_identity,
                reason_codes=() if status is core.OperandStatus.PASS else (f"DEV_{role.value}_FAILED",),
            )
            for index, role in enumerate(core.ADMISSION_ROLES)
        )
        facts.append(
            core.ActionCandidateFact(
                action_id=action_id,
                development_rank=ranks[action_id],
                target_response_positive=effect >= design.threshold("target-effect-min"),
                action_realization=realization,
                operands=operands,
            )
        )
    admitted = tuple(value for value in facts if value.admitted)
    selected = min(admitted, key=lambda value: (value.development_rank, value.action_id)).action_id if admitted else "hold"
    action_role = (
        core.ActionRole.REALIZATION_QUALIFIED
        if all(value.action_realization is core.ActionRole.REALIZATION_QUALIFIED for value in facts)
        else core.ActionRole.MISMATCH
    )
    receiver_role = core.ReceiverRole.NONLIMITING
    if any(value.operand(core.UniversalRole.RECEIVER_SINK).status is core.OperandStatus.FAIL for value in facts):
        receiver_role = core.ReceiverRole.SINK_LIMITING
    elif any(value.operand(core.UniversalRole.PRESERVATION).status is core.OperandStatus.FAIL for value in facts):
        receiver_role = core.ReceiverRole.PRESERVATION_LIMITING
    elif not admitted:
        receiver_role = core.ReceiverRole.TARGET_LIMITING
    restrictions = core.StructuralRestrictionProfile(
        preparation_role=(core.PreparationRole.DECISIVE if denominator is core.DenominatorStructure.INCOMPATIBLE else core.PreparationRole.NONDECISIVE_ON_CHART),
        history_role=(core.HistoryRole.QUOTIENT_ON_DECLARED_PAIR if history is core.HistoryClockQuotient.CLOSE_CLOSED else core.HistoryRole.TIMING_CONTROL_REQUIRED if history is core.HistoryClockQuotient.TIMING_CONFOUNDED else core.HistoryRole.RETAIN),
        action_role=action_role,
        receiver_role=receiver_role,
        numerical_role=(core.NumericalRole.QUALIFIED if denominator is core.DenominatorStructure.COMMON_LAW else core.NumericalRole.VIEW_LOCAL if denominator is core.DenominatorStructure.VIEW_LOCAL else core.NumericalRole.INTERACTION),
        boundary_role=(core.BoundaryRole.INTERIOR if support is core.SupportTransport.EQUALITY else core.BoundaryRole.SUPPORT_LOSS if support in {core.SupportTransport.BOUNDARY_LOSS, core.SupportTransport.OPPOSED} else core.BoundaryRole.BOUNDARY_LIMITING),
        internal_transfer_role=(core.InternalTransferRole.CLOSED if history is core.HistoryClockQuotient.CLOSE_CLOSED else core.InternalTransferRole.RELEVANT_DIRECTION if history is core.HistoryClockQuotient.CLOSE_DIVERGED else core.InternalTransferRole.UNRESOLVED),
        topology_mode_role=(core.TopologyModeRole.DECISIVE if design.target_slot.target_class_id == "graph-hybrid-system" else core.TopologyModeRole.INAPPLICABLE),
    )
    return (
        core.StructuralPredictionInput(
            input_id=f"{design.target_slot.target_slot_id}.development-input",
            ontology=core.structural_ontology(),
            donor_corpus=donor_corpus,
            target_slot=design.target_slot,
            compatibility=design.compatibility,
            denominator_evidence=denominator,
            history_evidence=history,
            support_evidence=support,
            restriction_evidence=restrictions,
            action_candidates=tuple(sorted(facts, key=lambda value: value.action_id)),
            hold_only_chart=False,
            deterministic_selector_available=True,
            observer_status=core.ObserverStatus.QUALIFIED,
            source_ready=True,
            receiver_available=True,
            prospective_units_available=bool(design.roster(StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION).unit_ids),
            hold_preservation_expected=True,
            checkpoint_receiver_close=history in {core.HistoryClockQuotient.CLOSE_CLOSED, core.HistoryClockQuotient.CLOSE_DIVERGED},
            prospective_validation_expectation=(core.DevelopmentControllerUseExpectation.VALIDATED if admitted else core.DevelopmentControllerUseExpectation.CONDITION_FALSE),
            missing_measurement_roles=(),
            required_new_role_ids=(),
            secondary_reason_codes=(),
            independent_development_unit_count=evidence.independent_unit_count,
            evidence_rung=EvidenceRung.LOCAL_LAW,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            raw_trajectory_values_present=False,
            native_threshold_values_present=False,
            target_admission_outcomes_accessed=False,
            target_prospective_validation_outcomes_accessed=False,
            post_reveal_diagnostics_accessed=False,
        ),
        selected,
    )


def decision_table_sha256() -> str:
    return sha256(
        b'categorical-structural-recurrence:frozen-ontology:categorical-only:noncompensating-controller-admission:hold-outside-admission'
    ).hexdigest()


__all__ = [
    'StructuralRecurrenceActionOutcome',
    'StructuralRecurrenceDevelopmentHandoff',
    'StructuralRecurrenceDonorCompilerFreeze',
    'StructuralRecurrenceNativeThreshold',
    'StructuralRecurrencePredictionIssue',
    'StructuralRecurrenceQualificationBranch',
    'StructuralRecurrenceSplitRoster',
    'StructuralRecurrenceSourceQualification',
    'StructuralRecurrenceStageEvidence',
    'StructuralRecurrenceTargetDesignFreeze',
    'StructuralRecurrenceTargetStage',
    'StructuralRecurrenceUnitObservation',
    "build_prediction_input",
    "decision_table_sha256",
]
