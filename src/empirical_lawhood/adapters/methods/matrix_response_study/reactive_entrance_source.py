"""Method-owned entrance projection and reactive entrance source qualification.

The simulator emits phase-space paths only.  This module applies the frozen
response-geometric conjunction, the robust-00 competing risk, duration gates,
same-driver numerical audit, and the single all-intent reactive entrance terminal.  It does
not fit or qualify a response law.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar, Sequence

import numpy as np
from scipy.stats import beta

from empirical_lawhood.adapters.composition.matrix_response_study.causal_intersection_residence_design import MatrixResponseCausalIntersectionResidenceStudyConfig
from empirical_lawhood.adapters.simulators.six_matrix_response.controlled_branch import SixMatrixResponseTransientControlledInvarianceActionLedger, SixMatrixResponseTransientControlledInvarianceBranchTrace, build_action_schedule
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import derive_probe_roster
from empirical_lawhood.adapters.simulators.six_matrix_response.reactive_entrance import REACTIVE_ENTRANCE_FINE_COUNT, REACTIVE_ENTRANCE_ROOT_COUNT, SixMatrixResponseReactiveEntranceHalf, SixMatrixResponseReactiveEntranceNumericalViewKind, SixMatrixResponseReactiveEntrancePersistedPrecursor, SixMatrixResponseReactiveEntrancePrecursorTerminal, SixMatrixResponseReactiveEntranceSourceConfig
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)

from .full_intersection_causal_authority import MatrixResponseCausalIntersectionResidenceResponseFactorSample, conservative_residence, evaluate_full_intersection_branch
from .reactive_nesting import MatrixResponseRNSourceCompatibilityReceipt, MatrixResponseRNSourceCompatibilityTerminal
from .shooting_committor import observe_state, rolling_labels


ENTRY_START_STEP = 256
ENTRY_END_STEP = 384
SOURCE_END_STEP = 512
ENTRY_CADENCE_STEPS = 16
DIRECT_DURATION_STEPS = (16, 32, 128)


class MatrixResponseReactiveEntranceProjectionTerminal(StrEnum):
    COMPLETE = "COMPLETE"
    NUMERICAL_INVALID = "NUMERICAL_INVALID"
    CUSTODY_INVALID = "CUSTODY_INVALID"


class MatrixResponseReactiveEntranceTerminal(StrEnum):
    QUALIFIED = "REACTIVE_ENTRANCE_SOURCE_QUALIFIED"
    TOO_SPARSE = "REACTIVE_ENTRANCE_SOURCE_TOO_SPARSE"
    NUMERICAL_INVALID = "REACTIVE_ENTRANCE_NUMERICAL_INVALID"
    CUSTODY_INVALID = "REACTIVE_ENTRANCE_CUSTODY_INVALID"
    PREREQUISITE_NONATTEMPT = "REACTIVE_ENTRANCE_PREREQUISITE_NONATTEMPT"


@dataclass(frozen=True, slots=True)
class MatrixResponseReactiveEntranceImplementationQualification(CanonicalRecord):
    """Pre-issue code qualification; it contains no reactive entrance outcome."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-reactive-entrance-implementation-qualification'

    qualification_id: str
    implementation_tree_sha256: str
    test_command_sha256: str
    test_output_sha256: str
    test_ids: tuple[str, ...]
    primary_restart_exact: bool
    half_restart_exact: bool
    hidden_conjugation_factor_invariant: bool
    hdf5_inventory_valid: bool
    hostile_inputs_refused: bool
    external_source_access_count: int
    rn0_outcome_access_count: int
    qualified: bool
    reason_codes: tuple[str, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        for name in (
            "implementation_tree_sha256",
            "test_command_sha256",
            "test_output_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.test_ids, field_name="test_ids", allow_empty=False)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected = bool(
            self.primary_restart_exact
            and self.half_restart_exact
            and self.hidden_conjugation_factor_invariant
            and self.hdf5_inventory_valid
            and self.hostile_inputs_refused
            and self.external_source_access_count == 0
            and self.rn0_outcome_access_count == 0
            and not self.reason_codes
        )
        if self.qualified != expected or self.grants_authority:
            raise ValueError("matrix response reactive entrance implementation qualification differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseReactiveEntranceMethodConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-reactive-entrance-method-config'

    config_id: str
    config_version: str
    source_config: ObjectIdentity
    compatibility_receipt: MatrixResponseRNSourceCompatibilityReceipt
    implementation_qualification: MatrixResponseReactiveEntranceImplementationQualification
    response_factor_config: MatrixResponseCausalIntersectionResidenceStudyConfig
    projection_task_prefix: str
    aggregate_task_id: str
    entry_start_step: int
    entry_end_step: int
    source_end_step: int
    receiver_cadence_steps: int
    direct_duration_steps: tuple[int, ...]
    hit_count_min: int
    residence_16_steps_count_min: int
    residence_16_steps_per_half_min: int
    numerical_dual_entry_min: int
    numerical_entry_time_difference_max: Decimal
    numerical_factor_agreement_min: Decimal
    confidence_level: Decimal
    source_instance_count: int
    physical_independent_unit_count: int
    numerical_view_count: int
    no_top_up: bool
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name in ("config_id", "projection_task_prefix", "aggregate_task_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.config_version)
        if self.source_config.object_schema != SixMatrixResponseReactiveEntranceSourceConfig.SCHEMA:
            raise ValueError("matrix response reactive entrance method binds another source config")
        if (
            self.compatibility_receipt.terminal
            is not MatrixResponseRNSourceCompatibilityTerminal.COMPATIBLE
            or self.compatibility_receipt.rn0_denominator_contribution != 0
            or not self.implementation_qualification.qualified
            or not isinstance(self.response_factor_config, MatrixResponseCausalIntersectionResidenceStudyConfig)
            or self.response_factor_config.q != 2
            or self.response_factor_config.primary_timestep != Decimal("0.001")
        ):
            raise ValueError("matrix response reactive entrance method prerequisite differs")
        if (
            (self.entry_start_step, self.entry_end_step, self.source_end_step)
            != (ENTRY_START_STEP, ENTRY_END_STEP, SOURCE_END_STEP)
            or self.receiver_cadence_steps != ENTRY_CADENCE_STEPS
            or self.direct_duration_steps != DIRECT_DURATION_STEPS
            or (self.hit_count_min, self.residence_16_steps_count_min, self.residence_16_steps_per_half_min)
            != (40, 32, 12)
            or self.numerical_dual_entry_min != 8
            or self.numerical_entry_time_difference_max != Decimal("0.032")
            or self.numerical_factor_agreement_min != Decimal("0.90")
            or self.confidence_level != Decimal("0.95")
            or (
                self.source_instance_count,
                self.physical_independent_unit_count,
                self.numerical_view_count,
            )
            != (1, REACTIVE_ENTRANCE_ROOT_COUNT, REACTIVE_ENTRANCE_FINE_COUNT)
            or not self.no_top_up
            or self.grants_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("matrix response reactive entrance method fixed gate differs")
        for name in (
            "numerical_entry_time_difference_max",
            "numerical_factor_agreement_min",
            "confidence_level",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class MatrixResponseReactiveEntranceProjection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-reactive-entrance-projection'

    projection_id: str
    source_result: ObjectIdentity
    source_config: ObjectIdentity
    method_config: ObjectIdentity
    slot_id: str
    slot_index: int
    half: SixMatrixResponseReactiveEntranceHalf
    physical_independent_unit_id: str
    acquisition_group_id: str
    scientific_view_id: str
    numerical_view_kind: SixMatrixResponseReactiveEntranceNumericalViewKind
    factor_samples: tuple[MatrixResponseCausalIntersectionResidenceResponseFactorSample, ...]
    robust_00_step: int | None
    first_entry_step: int | None
    entrance_hit: bool
    residence_16_steps: bool
    residence_32_steps: bool
    residence_128_steps: bool
    full_total_residence: Decimal
    full_longest_residence: Decimal
    entry_positions_sha256: str | None
    entry_momenta_sha256: str | None
    terminal: MatrixResponseReactiveEntranceProjectionTerminal
    custody_valid: bool
    technically_valid: bool
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name in (
            "projection_id",
            "slot_id",
            "physical_independent_unit_id",
            "acquisition_group_id",
            "scientific_view_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        for name in ("full_total_residence", "full_longest_residence"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.entry_positions_sha256 is not None:
            validate_sha256(self.entry_positions_sha256, field_name="entry_positions_sha256")
        if self.entry_momenta_sha256 is not None:
            validate_sha256(self.entry_momenta_sha256, field_name="entry_momenta_sha256")
        valid_steps = tuple(range(ENTRY_START_STEP, SOURCE_END_STEP + 1, ENTRY_CADENCE_STEPS))
        if self.factor_samples and tuple(value.parent_step for value in self.factor_samples) != valid_steps:
            raise ValueError("matrix response reactive entrance factor ledger differs")
        if self.first_entry_step is not None and self.first_entry_step not in range(
            ENTRY_START_STEP,
            ENTRY_END_STEP + 1,
            ENTRY_CADENCE_STEPS,
        ):
            raise ValueError("matrix response reactive entrance entry lies outside its causal grid")
        expected_hit = bool(
            self.first_entry_step is not None
            and (self.robust_00_step is None or self.first_entry_step < self.robust_00_step)
        )
        if self.entrance_hit != expected_hit:
            raise ValueError("matrix response reactive entrance competing-risk entry differs")
        if self.factor_samples:
            candidate = next(
                (
                    value.parent_step
                    for value in self.factor_samples
                    if value.parent_step <= ENTRY_END_STEP and value.full_intersection_pass
                ),
                None,
            )
            expected_entry = (
                candidate
                if candidate is not None
                and (self.robust_00_step is None or candidate < self.robust_00_step)
                else None
            )
            if self.first_entry_step != expected_entry:
                raise ValueError("matrix response reactive entrance projection entry differs from its factor ledger")
            expected_durations = (
                (False, False, False)
                if expected_entry is None
                else tuple(
                    _duration(self.factor_samples, entry_step=expected_entry, steps=steps)
                    for steps in DIRECT_DURATION_STEPS
                )
            )
            if (self.residence_16_steps, self.residence_32_steps, self.residence_128_steps) != expected_durations:
                raise ValueError("matrix response reactive entrance projection duration differs from its factor ledger")
            flags = (
                ()
                if expected_entry is None
                else tuple(
                    value.full_intersection_pass
                    for value in self.factor_samples
                    if value.parent_step >= expected_entry
                )
            )
            total, longest = conservative_residence(flags, cadence_time=0.016)
            if (
                self.full_total_residence != _decimal(total)
                or self.full_longest_residence != _decimal(longest)
            ):
                raise ValueError("matrix response reactive entrance projection residence differs from its factor ledger")
        elif self.first_entry_step is not None:
            raise ValueError("matrix response reactive entrance entry lacks a factor ledger")
        if not self.entrance_hit and any((self.residence_16_steps, self.residence_32_steps, self.residence_128_steps)):
            raise ValueError("matrix response reactive entrance duration cannot replace a nonentry")
        if self.residence_32_steps and not self.residence_16_steps or self.residence_128_steps and not self.residence_32_steps:
            raise ValueError("matrix response reactive entrance duration flags are not nested")
        if self.entrance_hit != bool(self.entry_positions_sha256 and self.entry_momenta_sha256):
            raise ValueError("matrix response reactive entrance entry checkpoint custody differs")
        if self.terminal is MatrixResponseReactiveEntranceProjectionTerminal.COMPLETE:
            if not (self.custody_valid and self.technically_valid) or self.reason_codes:
                raise ValueError("complete matrix response reactive entrance projection is invalid")
        elif not self.reason_codes:
            raise ValueError("invalid matrix response reactive entrance projection requires reasons")
        if (
            self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("matrix response reactive entrance projection visibility differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseReactiveEntranceBinomialInterval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-reactive-entrance-binomial-interval'

    interval_id: str
    endpoint_id: str
    successes: int
    trials: int
    estimate: Decimal
    lower: Decimal
    upper: Decimal
    confidence_level: Decimal
    method_id: str

    def __post_init__(self) -> None:
        for name in ("interval_id", "endpoint_id", "method_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if not 0 <= self.successes <= self.trials or self.trials < 1:
            raise ValueError("matrix response reactive entrance binomial counts differ")
        for name in ("estimate", "lower", "upper", "confidence_level"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if not self.lower <= self.estimate <= self.upper <= Decimal(1):
            raise ValueError("matrix response reactive entrance binomial interval ordering differs")
        if self.confidence_level != Decimal("0.95") or self.method_id != "clopper-pearson-exact":
            raise ValueError("matrix response reactive entrance interval method differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseReactiveEntranceNumericalAudit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-reactive-entrance-numerical-audit'

    audit_id: str
    planned_pair_count: int
    valid_pair_count: int
    dual_entry_count: int
    primary_only_entry_count: int
    half_only_entry_count: int
    maximum_dual_entry_time_difference: Decimal | None
    factor_comparison_count: int
    factor_agreement_count: int
    factor_agreement: Decimal
    minimum_dual_entry_count: int
    maximum_entry_time_difference: Decimal
    minimum_factor_agreement: Decimal
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        for name in (
            "planned_pair_count",
            "valid_pair_count",
            "dual_entry_count",
            "primary_only_entry_count",
            "half_only_entry_count",
            "factor_comparison_count",
            "factor_agreement_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.planned_pair_count != REACTIVE_ENTRANCE_FINE_COUNT or not (
            self.dual_entry_count
            + self.primary_only_entry_count
            + self.half_only_entry_count
            <= self.valid_pair_count
            <= self.planned_pair_count
        ):
            raise ValueError("matrix response reactive entrance numerical pair accounting differs")
        if not 0 <= self.factor_agreement_count <= self.factor_comparison_count:
            raise ValueError("matrix response reactive entrance factor agreement counts differ")
        for name in (
            "factor_agreement",
            "maximum_entry_time_difference",
            "minimum_factor_agreement",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.maximum_dual_entry_time_difference is not None:
            validate_decimal(
                self.maximum_dual_entry_time_difference,
                field_name="maximum_dual_entry_time_difference",
                minimum=Decimal(0),
            )
        if self.passed != (not self.reason_codes):
            raise ValueError("matrix response reactive entrance numerical audit disposition differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseReactiveEntranceAggregate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-reactive-entrance-aggregate'

    aggregate_id: str
    source_config: ObjectIdentity
    method_config: ObjectIdentity
    compatibility_receipt: ObjectIdentity
    implementation_qualification: ObjectIdentity
    projection_identities: tuple[ObjectIdentity, ...]
    source_instance_count: int
    physical_independent_unit_count: int
    primary_projection_count: int
    numerical_projection_count: int
    complete_primary_count: int
    complete_numerical_count: int
    entrance_hit_count: int
    residence_16_steps_count: int
    residence_32_steps_count: int
    residence_128_steps_count: int
    first_half_residence_16_steps_count: int
    second_half_residence_16_steps_count: int
    intervals: tuple[MatrixResponseReactiveEntranceBinomialInterval, ...]
    numerical_audit: MatrixResponseReactiveEntranceNumericalAudit
    no_top_up: bool
    terminal: MatrixResponseReactiveEntranceTerminal
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.aggregate_id, field_name="aggregate_id")
        require_sorted_unique_ids(
            self.projection_identities,
            attribute="object_id",
            field_name="projection_identities",
        )
        require_sorted_unique_ids(self.intervals, attribute="interval_id", field_name="intervals")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            (self.source_instance_count, self.physical_independent_unit_count)
            != (1, REACTIVE_ENTRANCE_ROOT_COUNT)
            or not 0 <= self.primary_projection_count <= REACTIVE_ENTRANCE_ROOT_COUNT
            or not 0 <= self.numerical_projection_count <= REACTIVE_ENTRANCE_FINE_COUNT
            or len(self.projection_identities)
            != self.primary_projection_count + self.numerical_projection_count
            or len(self.intervals) != 4
            or not self.no_top_up
            or self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or self.grants_authority
        ):
            raise ValueError("matrix response reactive entrance aggregate contract differs")
        if self.terminal is MatrixResponseReactiveEntranceTerminal.QUALIFIED and (
            self.primary_projection_count != REACTIVE_ENTRANCE_ROOT_COUNT
            or self.numerical_projection_count != REACTIVE_ENTRANCE_FINE_COUNT
            or self.complete_primary_count != REACTIVE_ENTRANCE_ROOT_COUNT
            or self.complete_numerical_count != REACTIVE_ENTRANCE_FINE_COUNT
        ):
            raise ValueError("qualified matrix response reactive entrance aggregate lacks its complete roster")
        if (self.terminal is MatrixResponseReactiveEntranceTerminal.QUALIFIED) != (not self.reason_codes):
            raise ValueError("matrix response reactive entrance aggregate terminal/reasons differ")


def _decimal(value: float) -> Decimal:
    return Decimal(repr(float(value)))


def _hold_trace(
    *, persisted: SixMatrixResponseReactiveEntrancePersistedPrecursor, source: SixMatrixResponseReactiveEntranceSourceConfig
) -> SixMatrixResponseTransientControlledInvarianceBranchTrace:
    result = persisted.result
    multiplier = (
        1
        if result.numerical_view_kind is SixMatrixResponseReactiveEntranceNumericalViewKind.PRIMARY
        else 2
    )
    view = source.primary_view if multiplier == 1 else source.half_view
    schedule = build_action_schedule(
        action_word="hold",
        branch_start_step=0,
        trigger_parent_step=None,
        total_primary_steps=result.completed_steps,
        baseline_x=float(source.target_alpha_tilde_x),
        baseline_y=float(source.target_alpha_tilde_y),
        timestep=float(view.timestep),
    )
    ledger = SixMatrixResponseTransientControlledInvarianceActionLedger(
        ledger_id=f"ledger.{result.result_id}",
        action_word="hold",
        trigger_parent_step=None,
        requested_sha256=schedule.requested_sha256,
        accepted_sha256=schedule.requested_sha256,
        applied_sha256=schedule.requested_sha256,
        realized_sha256=schedule.requested_sha256,
        maximum_excursion=Decimal(0),
        maximum_increment=Decimal(0),
        total_variation=Decimal(0),
        squared_action_energy=Decimal(0),
        generalized_absolute_work=Decimal(0),
        pulse_count=0,
        exact_baseline_return=True,
        clipped=False,
        valid=True,
        reason_codes=(),
    )
    cadence = ENTRY_CADENCE_STEPS * multiplier
    receivers = tuple(
        persisted.states[step]
        for step in range(cadence, result.completed_steps + 1, cadence)
    )
    return SixMatrixResponseTransientControlledInvarianceBranchTrace(
        branch_id=f"matrix-response-reactive-entrance.trace.r{result.slot_index:04d}.{result.numerical_view_kind.value.lower()}",
        block_index=result.slot_index,
        numerical_view_id=(source.primary_view.view_id if multiplier == 1 else source.half_view.view_id),
        parent_step_multiplier=multiplier,
        start_state=persisted.states[0],
        final_state=persisted.states[-1],
        receiver_states=receivers,
        y_path=np.ascontiguousarray(
            np.stack([value.positions[1] for value in persisted.states]), dtype="<c16"
        ),
        noise_seed_sha256=result.noise.source_seed_sha256,
        noise_block_sha256=result.noise.realized_innovation_sha256,
        action_ledger=ledger,
        valid=True,
        reason_codes=(),
    )


def _duration(
    samples: Sequence[MatrixResponseCausalIntersectionResidenceResponseFactorSample], *, entry_step: int, steps: int
) -> bool:
    by_step = {value.parent_step: value.full_intersection_pass for value in samples}
    return all(
        by_step.get(step, False)
        for step in range(entry_step, entry_step + steps + 1, ENTRY_CADENCE_STEPS)
    )


def project_matrix_response_study_reactive_entrance_precursor(
    *,
    persisted: SixMatrixResponseReactiveEntrancePersistedPrecursor,
    source_config: SixMatrixResponseReactiveEntranceSourceConfig,
    method_config: MatrixResponseReactiveEntranceMethodConfig,
) -> MatrixResponseReactiveEntranceProjection:
    """Apply the frozen direct entrance rule to one persisted source path."""

    result = persisted.result
    source_identity = ObjectIdentity.from_record(source_config.config_id, source_config)
    method_identity = ObjectIdentity.from_record(method_config.config_id, method_config)
    slot = source_config.slot(result.slot_index)
    reasons: set[str] = set()
    if (
        method_config.source_config != source_identity
        or result.source_config != source_identity
        or result.slot_id != slot.slot_id
        or result.physical_independent_unit_id != slot.physical_independent_unit_id
    ):
        reasons.add("source-custody-mismatch")
    if result.terminal is not SixMatrixResponseReactiveEntrancePrecursorTerminal.COMPLETED:
        reasons.add("source-numerical-invalid")
    if result.completed_steps != result.total_steps:
        reasons.add("source-path-incomplete")
    factor_samples: tuple[MatrixResponseCausalIntersectionResidenceResponseFactorSample, ...] = ()
    robust_00_step: int | None = None
    first_entry_step: int | None = None
    entry_positions: str | None = None
    entry_momenta: str | None = None
    residence_16_steps = residence_32_steps = residence_128_steps = False
    total = longest = 0.0
    if not reasons:
        trace = _hold_trace(persisted=persisted, source=source_config)
        roster = derive_probe_roster(
            config_fingerprint=(
                method_config.response_factor_config.probe_roster_config_fingerprint
            ),
            rule_id=method_config.response_factor_config.probe_seed_rule_id,
            scientific_seed=int(
                method_config.response_factor_config.probe_scientific_input.scientific_seed_sha256, 16
            ),
        )
        evaluation = evaluate_full_intersection_branch(
            trace=trace,
            intent_word="hold",
            member=source_config.member,
            roster=roster,
            config=method_config.response_factor_config,
            assessment_start_step=ENTRY_START_STEP,
            assessment_end_step=SOURCE_END_STEP,
        )
        factor_samples = evaluation.samples
        if not evaluation.outcome.technically_valid:
            reasons.update(evaluation.outcome.reason_codes or ("factor-evaluation-invalid",))
        multiplier = trace.parent_step_multiplier
        observations = tuple(
            observe_state(
                observation_id=(
                    f"matrix-response-reactive-entrance.observation.r{result.slot_index:04d}."
                    f"{state.step_index // multiplier:04d}"
                ),
                local_step=state.step_index // multiplier,
                local_time=(state.step_index // multiplier)
                * float(method_config.response_factor_config.primary_timestep),
                state=state,
                member=source_config.member,
                config=method_config.response_factor_config,
            )
            for state in trace.receiver_states
        )
        labels = rolling_labels(observations, config=method_config.response_factor_config)
        robust_00_step = next(
            (value.endpoint_step for value in labels if value.label == "00"), None
        )
        candidate = next(
            (
                value.parent_step
                for value in factor_samples
                if value.parent_step <= ENTRY_END_STEP and value.full_intersection_pass
            ),
            None,
        )
        if candidate is not None and (robust_00_step is None or candidate < robust_00_step):
            first_entry_step = candidate
            residence_16_steps = _duration(factor_samples, entry_step=candidate, steps=16)
            residence_32_steps = _duration(factor_samples, entry_step=candidate, steps=32)
            residence_128_steps = _duration(factor_samples, entry_step=candidate, steps=128)
            offset = multiplier * candidate
            entry_state = persisted.states[offset]
            entry_positions = sha256(entry_state.positions.tobytes()).hexdigest()
            entry_momenta = sha256(entry_state.momenta.tobytes()).hexdigest()
            suffix = tuple(
                value.full_intersection_pass
                for value in factor_samples
                if value.parent_step >= candidate
            )
            total, longest = conservative_residence(
                suffix,
                cadence_time=float(method_config.response_factor_config.primary_timestep)
                * ENTRY_CADENCE_STEPS,
            )
    custody_valid = "source-custody-mismatch" not in reasons
    technically_valid = not {
        "source-numerical-invalid",
        "source-path-incomplete",
        "factor-evaluation-invalid",
    }.intersection(reasons)
    terminal = (
        MatrixResponseReactiveEntranceProjectionTerminal.CUSTODY_INVALID
        if not custody_valid
        else (
            MatrixResponseReactiveEntranceProjectionTerminal.NUMERICAL_INVALID
            if not technically_valid
            else MatrixResponseReactiveEntranceProjectionTerminal.COMPLETE
        )
    )
    return MatrixResponseReactiveEntranceProjection(
        projection_id=f"matrix-response-reactive-entrance.projection.{result.scientific_view_id}",
        source_result=ObjectIdentity.from_record(result.result_id, result),
        source_config=source_identity,
        method_config=method_identity,
        slot_id=slot.slot_id,
        slot_index=slot.slot_index,
        half=slot.half,
        physical_independent_unit_id=slot.physical_independent_unit_id,
        acquisition_group_id=result.acquisition_group_id,
        scientific_view_id=result.scientific_view_id,
        numerical_view_kind=result.numerical_view_kind,
        factor_samples=factor_samples,
        robust_00_step=robust_00_step,
        first_entry_step=first_entry_step,
        entrance_hit=first_entry_step is not None,
        residence_16_steps=residence_16_steps,
        residence_32_steps=residence_32_steps,
        residence_128_steps=residence_128_steps,
        full_total_residence=_decimal(total),
        full_longest_residence=_decimal(longest),
        entry_positions_sha256=entry_positions,
        entry_momenta_sha256=entry_momenta,
        terminal=terminal,
        custody_valid=custody_valid,
        technically_valid=technically_valid,
        reason_codes=tuple(sorted(reasons)),
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def _interval(endpoint: str, successes: int, trials: int) -> MatrixResponseReactiveEntranceBinomialInterval:
    alpha = 0.05
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return MatrixResponseReactiveEntranceBinomialInterval(
        interval_id=f"matrix-response-reactive-entrance.interval.{endpoint}",
        endpoint_id=endpoint,
        successes=successes,
        trials=trials,
        estimate=_decimal(successes / trials),
        lower=_decimal(lower),
        upper=_decimal(upper),
        confidence_level=Decimal("0.95"),
        method_id="clopper-pearson-exact",
    )


def _numerical_audit(
    *, primary: dict[int, MatrixResponseReactiveEntranceProjection], fine: dict[int, MatrixResponseReactiveEntranceProjection]
) -> MatrixResponseReactiveEntranceNumericalAudit:
    valid_slots = tuple(
        sorted(
            index
            for index in fine
            if primary[index].terminal is MatrixResponseReactiveEntranceProjectionTerminal.COMPLETE
            and fine[index].terminal is MatrixResponseReactiveEntranceProjectionTerminal.COMPLETE
        )
    )
    dual = tuple(index for index in valid_slots if primary[index].entrance_hit and fine[index].entrance_hit)
    primary_only = tuple(
        index for index in valid_slots if primary[index].entrance_hit and not fine[index].entrance_hit
    )
    half_only = tuple(
        index for index in valid_slots if fine[index].entrance_hit and not primary[index].entrance_hit
    )
    differences: list[float] = []
    for index in dual:
        primary_step = primary[index].first_entry_step
        fine_step = fine[index].first_entry_step
        if primary_step is None or fine_step is None:
            raise AssertionError("dual matrix response reactive entrance entrance lacks an entry step")
        differences.append(abs(primary_step - fine_step) / 1000)
    comparisons = 0
    agreements = 0
    for index in valid_slots:
        left = primary[index].factor_samples
        right = fine[index].factor_samples
        if tuple(value.parent_step for value in left) != tuple(value.parent_step for value in right):
            continue
        comparisons += len(left)
        agreements += sum(
            first.full_intersection_pass == second.full_intersection_pass
            for first, second in zip(left, right, strict=True)
        )
    agreement = agreements / comparisons if comparisons else 0.0
    reasons: set[str] = set()
    if len(valid_slots) != REACTIVE_ENTRANCE_FINE_COUNT:
        reasons.add("numerical-pair-roster-incomplete")
    if len(dual) < 8:
        reasons.add("dual-entry-count-below-floor")
    if differences and max(differences) > 0.032 + 1e-15:
        reasons.add("dual-entry-time-discordant")
    if agreement < 0.90 - 1e-15:
        reasons.add("factor-classification-agreement-below-floor")
    if half_only:
        reasons.add("half-step-only-false-admission")
    return MatrixResponseReactiveEntranceNumericalAudit(
        audit_id="matrix-response-reactive-entrance.numerical-audit",
        planned_pair_count=REACTIVE_ENTRANCE_FINE_COUNT,
        valid_pair_count=len(valid_slots),
        dual_entry_count=len(dual),
        primary_only_entry_count=len(primary_only),
        half_only_entry_count=len(half_only),
        maximum_dual_entry_time_difference=(
            None if not differences else _decimal(max(differences))
        ),
        factor_comparison_count=comparisons,
        factor_agreement_count=agreements,
        factor_agreement=_decimal(agreement),
        minimum_dual_entry_count=8,
        maximum_entry_time_difference=Decimal("0.032"),
        minimum_factor_agreement=Decimal("0.90"),
        passed=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


def reduce_matrix_response_study_reactive_entrance_source(
    *,
    source_config: SixMatrixResponseReactiveEntranceSourceConfig,
    method_config: MatrixResponseReactiveEntranceMethodConfig,
    projections: tuple[MatrixResponseReactiveEntranceProjection, ...],
) -> MatrixResponseReactiveEntranceAggregate:
    """Close the exact 512+128 roster once with noncompensating precedence."""

    source_identity = ObjectIdentity.from_record(source_config.config_id, source_config)
    method_identity = ObjectIdentity.from_record(method_config.config_id, method_config)
    primary_values = tuple(
        value
        for value in projections
        if value.numerical_view_kind is SixMatrixResponseReactiveEntranceNumericalViewKind.PRIMARY
    )
    fine_values = tuple(
        value
        for value in projections
        if value.numerical_view_kind is SixMatrixResponseReactiveEntranceNumericalViewKind.SAME_DRIVER_HALF
    )
    primary = {value.slot_index: value for value in primary_values}
    fine = {value.slot_index: value for value in fine_values}
    custody_reasons: set[str] = set()
    if (
        method_config.source_config != source_identity
        or len(primary_values) != len(primary)
        or len(fine_values) != len(fine)
        or set(primary) != set(range(REACTIVE_ENTRANCE_ROOT_COUNT))
        or set(fine)
        != {value.slot_index for value in source_config.slots if value.has_fine_view}
        or any(
            value.source_config != source_identity or value.method_config != method_identity
            for value in projections
        )
    ):
        custody_reasons.add("projection-roster-or-parent-mismatch")
    custody_reasons.update(
        "projection-custody-invalid"
        for value in projections
        if value.terminal is MatrixResponseReactiveEntranceProjectionTerminal.CUSTODY_INVALID
    )
    numerical_projection_reasons = {
        "projection-numerical-invalid"
        for value in projections
        if value.terminal is MatrixResponseReactiveEntranceProjectionTerminal.NUMERICAL_INVALID
    }
    audit = _numerical_audit(primary=primary, fine=fine) if not custody_reasons else MatrixResponseReactiveEntranceNumericalAudit(
        audit_id="matrix-response-reactive-entrance.numerical-audit",
        planned_pair_count=REACTIVE_ENTRANCE_FINE_COUNT,
        valid_pair_count=0,
        dual_entry_count=0,
        primary_only_entry_count=0,
        half_only_entry_count=0,
        maximum_dual_entry_time_difference=None,
        factor_comparison_count=0,
        factor_agreement_count=0,
        factor_agreement=Decimal(0),
        minimum_dual_entry_count=8,
        maximum_entry_time_difference=Decimal("0.032"),
        minimum_factor_agreement=Decimal("0.90"),
        passed=False,
        reason_codes=("numerical-audit-not-evaluable-after-custody-failure",),
    )
    complete_primary = tuple(
        value for value in primary_values if value.terminal is MatrixResponseReactiveEntranceProjectionTerminal.COMPLETE
    )
    complete_fine = tuple(
        value for value in fine_values if value.terminal is MatrixResponseReactiveEntranceProjectionTerminal.COMPLETE
    )
    hit = sum(value.entrance_hit for value in complete_primary)
    residence_16_steps = sum(value.residence_16_steps for value in complete_primary)
    residence_32_steps = sum(value.residence_32_steps for value in complete_primary)
    residence_128_steps = sum(value.residence_128_steps for value in complete_primary)
    first_residence_16_steps = sum(
        value.residence_16_steps for value in complete_primary if value.half is SixMatrixResponseReactiveEntranceHalf.FIRST
    )
    second_residence_16_steps = sum(
        value.residence_16_steps for value in complete_primary if value.half is SixMatrixResponseReactiveEntranceHalf.SECOND
    )
    prerequisite_reasons: set[str] = set()
    if (
        method_config.compatibility_receipt.terminal
        is not MatrixResponseRNSourceCompatibilityTerminal.COMPATIBLE
        or not method_config.implementation_qualification.qualified
    ):
        prerequisite_reasons.add("source-prerequisite-not-qualified")
    sparse_reasons: set[str] = set()
    if hit < method_config.hit_count_min:
        sparse_reasons.add("entry-count-below-floor")
    if residence_16_steps < method_config.residence_16_steps_count_min:
        sparse_reasons.add("residence_16_steps-count-below-floor")
    if first_residence_16_steps < method_config.residence_16_steps_per_half_min:
        sparse_reasons.add("first-half-residence_16_steps-count-below-floor")
    if second_residence_16_steps < method_config.residence_16_steps_per_half_min:
        sparse_reasons.add("second-half-residence_16_steps-count-below-floor")
    hard_audit_reasons = set(audit.reason_codes) - {"dual-entry-count-below-floor"}
    if prerequisite_reasons:
        terminal = MatrixResponseReactiveEntranceTerminal.PREREQUISITE_NONATTEMPT
        reasons = prerequisite_reasons
    elif custody_reasons:
        terminal = MatrixResponseReactiveEntranceTerminal.CUSTODY_INVALID
        reasons = custody_reasons
    elif numerical_projection_reasons or hard_audit_reasons:
        terminal = MatrixResponseReactiveEntranceTerminal.NUMERICAL_INVALID
        reasons = numerical_projection_reasons | hard_audit_reasons
    elif sparse_reasons:
        terminal = MatrixResponseReactiveEntranceTerminal.TOO_SPARSE
        reasons = sparse_reasons
    elif "dual-entry-count-below-floor" in audit.reason_codes:
        terminal = MatrixResponseReactiveEntranceTerminal.NUMERICAL_INVALID
        reasons = {"dual-entry-count-below-floor"}
    else:
        terminal = MatrixResponseReactiveEntranceTerminal.QUALIFIED
        reasons = set()
    intervals = tuple(
        _interval(endpoint, count, REACTIVE_ENTRANCE_ROOT_COUNT)
        for endpoint, count in (
            ("entrance-hit", hit),
            ("residence-16-steps", residence_16_steps),
            ("residence-32-steps", residence_32_steps),
            ("residence-128-steps", residence_128_steps),
        )
    )
    identities = tuple(
        sorted(
            (ObjectIdentity.from_record(value.projection_id, value) for value in projections),
            key=lambda value: value.object_id,
        )
    )
    return MatrixResponseReactiveEntranceAggregate(
        aggregate_id="matrix-response-reactive-entrance.source-aggregate.law-development",
        source_config=source_identity,
        method_config=method_identity,
        compatibility_receipt=ObjectIdentity.from_record(
            method_config.compatibility_receipt.receipt_id,
            method_config.compatibility_receipt,
        ),
        implementation_qualification=ObjectIdentity.from_record(
            method_config.implementation_qualification.qualification_id,
            method_config.implementation_qualification,
        ),
        projection_identities=identities,
        source_instance_count=1,
        physical_independent_unit_count=REACTIVE_ENTRANCE_ROOT_COUNT,
        primary_projection_count=len(primary_values),
        numerical_projection_count=len(fine_values),
        complete_primary_count=len(complete_primary),
        complete_numerical_count=len(complete_fine),
        entrance_hit_count=hit,
        residence_16_steps_count=residence_16_steps,
        residence_32_steps_count=residence_32_steps,
        residence_128_steps_count=residence_128_steps,
        first_half_residence_16_steps_count=first_residence_16_steps,
        second_half_residence_16_steps_count=second_residence_16_steps,
        intervals=tuple(sorted(intervals, key=lambda value: value.interval_id)),
        numerical_audit=audit,
        no_top_up=True,
        terminal=terminal,
        reason_codes=tuple(sorted(reasons)),
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        grants_authority=False,
    )


__all__ = [
    "DIRECT_DURATION_STEPS",
    "ENTRY_CADENCE_STEPS",
    "ENTRY_END_STEP",
    "ENTRY_START_STEP",
    "SOURCE_END_STEP",
    'MatrixResponseReactiveEntranceAggregate',
    'MatrixResponseReactiveEntranceBinomialInterval',
    'MatrixResponseReactiveEntranceImplementationQualification',
    'MatrixResponseReactiveEntranceMethodConfig',
    'MatrixResponseReactiveEntranceNumericalAudit',
    'MatrixResponseReactiveEntranceProjectionTerminal',
    'MatrixResponseReactiveEntranceProjection',
    'MatrixResponseReactiveEntranceTerminal',
    'project_matrix_response_study_reactive_entrance_precursor',
    'reduce_matrix_response_study_reactive_entrance_source',
]
