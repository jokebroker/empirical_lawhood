"""Fresh-process factor evaluation from persisted Matrix preparation scheduling trajectory bytes."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.composition.matrix_response_study.causal_intersection_residence_design import MatrixResponseCausalIntersectionResidenceStudyConfig
from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseModelFamilyMember
from empirical_lawhood.adapters.simulators.six_matrix_response.model import SixMatrixState
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import SixMatrixResponseTransientControlledInvarianceProbeRoster, derive_probe_roster, heat_predict_probe, normalized_increment_loss, propagate_passive_probes, radius_only_operator, traceless_operator
from empirical_lawhood.adapters.simulators.six_matrix_response.preparation_schedule import SixMatrixResponsePreparationWindowSchedulingPersistedTrajectory, SixMatrixResponsePreparationWindowSchedulingScheduleTerminal
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)

from .prospective_window_scheduling import MatrixResponsePreparationWindowSchedulingFactorOperands, MatrixResponsePreparationWindowSchedulingFactorRecordDisposition, MatrixResponsePreparationWindowSchedulingFactorSample, MatrixResponsePreparationWindowSchedulingFactorThresholds, MatrixResponsePreparationWindowSchedulingTrajectoryFactorRecord, MatrixResponsePreparationWindowSchedulingTrajectoryOutcome, evaluate_factor_sample, reduce_trajectory_window
from .shooting_committor import observe_state, rolling_labels


@dataclass(frozen=True, slots=True)
class MatrixResponsePreparationWindowSchedulingFactorEvaluationConfig(CanonicalRecord):
    """Only the frozen factor/probe choices; no causal intersection residence pulse semantics are carried."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-preparation-window-scheduling-factor-evaluation-config'

    config_id: str
    config_version: str
    predecessor_factor_config: ObjectIdentity
    predecessor_factor_record: MatrixResponseCausalIntersectionResidenceStudyConfig
    family_member: ObjectIdentity
    thresholds: MatrixResponsePreparationWindowSchedulingFactorThresholds
    q: int
    primary_timestep: Decimal
    receiver_cadence_steps: int
    rolling_window_samples: int
    first_eligible_post_arrival_step: int
    last_post_arrival_step: int
    heldout_probe_field_indices: tuple[int, ...]
    probe_kappas: tuple[Decimal, ...]
    probe_roster_field_ids: tuple[str, ...]
    probe_roster_seed_sha256: str
    probe_numeric_floor: Decimal
    probe_hermiticity_residual_max: Decimal
    probe_trace_residual_max: Decimal
    probe_identity_drift_max: Decimal
    probe_norm_increase_max: Decimal
    probe_forecast_lag_steps: int
    evaluator_rule_id: str
    grants_authority: bool = False

    def __post_init__(self) -> None:
        for name in ("config_id", "evaluator_rule_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.config_version)
        predecessor = self.predecessor_factor_record
        if self.predecessor_factor_config != ObjectIdentity.from_record(
            predecessor.config_id, predecessor
        ):
            raise ValueError("preparation scheduling factor config requires its supplied causal intersection residence factor source")
        if self.family_member.object_schema != SixMatrixResponseModelFamilyMember.SCHEMA:
            raise ValueError("preparation scheduling factor config requires the exact model-family member")
        if (
            self.q,
            self.primary_timestep,
            self.receiver_cadence_steps,
            self.rolling_window_samples,
            self.first_eligible_post_arrival_step,
            self.last_post_arrival_step,
            self.heldout_probe_field_indices,
            self.probe_kappas,
            self.probe_forecast_lag_steps,
        ) != (
            2,
            Decimal("0.001"),
            16,
            16,
            256,
            768,
            (6, 7, 8, 9, 10, 11),
            (Decimal("0.25"), Decimal("0.5"), Decimal("1")),
            32,
        ):
            raise ValueError("preparation scheduling factor timing/probe design differs")
        if self.thresholds != MatrixResponsePreparationWindowSchedulingFactorThresholds(
            thresholds_id="matrix-preparation-scheduling.factor-thresholds.from-matrix-causal-intersection-residence",
            phi_min=Decimal("0.35"),
            phi_max=Decimal("0.95"),
            closure_ratio_max=Decimal("0.30"),
            kernel_band_ratio_max=Decimal("0.25"),
            persistence_pass_count=12,
            response_geometry_loss_max=Decimal("0.25"),
            response_radius_ratio_max=Decimal("0.75"),
        ):
            raise ValueError("preparation scheduling factor thresholds are not the exact immutable causal intersection residence projection")
        require_sorted_unique_strings(
            self.probe_roster_field_ids,
            field_name="probe_roster_field_ids",
            allow_empty=False,
        )
        if len(self.probe_roster_field_ids) != 12:
            raise ValueError("preparation scheduling factor config requires the complete 12-field roster")
        validate_sha256(self.probe_roster_seed_sha256, field_name="probe_roster_seed_sha256")
        for name in (
            "probe_numeric_floor", "probe_hermiticity_residual_max",
            "probe_trace_residual_max", "probe_identity_drift_max",
            "probe_norm_increase_max",
        ):
            value = getattr(self, name)
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value == 0:
                raise ValueError("preparation scheduling probe bound must be positive")
        if (
            self.probe_numeric_floor,
            self.probe_hermiticity_residual_max,
            self.probe_trace_residual_max,
            self.probe_identity_drift_max,
            self.probe_norm_increase_max,
        ) != (
            Decimal("1e-12"),
            Decimal("1e-10"),
            Decimal("1e-10"),
            Decimal("1e-10"),
            Decimal("1e-10"),
        ):
            raise ValueError("preparation scheduling factor probe invariant bounds differ from causal intersection residence")
        expected_roster = derive_probe_roster(
            config_fingerprint=predecessor.probe_roster_config_fingerprint,
            rule_id=predecessor.probe_seed_rule_id,
            scientific_seed=int(predecessor.probe_scientific_input.scientific_seed_sha256, 16),
        )
        if (
            self.probe_roster_field_ids != expected_roster.field_ids
            or self.probe_roster_seed_sha256 != expected_roster.seed_sha256
            or self.family_member.object_id != predecessor.member_id
            or self.family_member.object_fingerprint != predecessor.member_fingerprint
            or self.evaluator_rule_id != "matrix-preparation-scheduling.pure-factor-evaluator.from-causal-intersection-residence"
        ):
            raise ValueError("preparation scheduling factor member/roster/rule identity differs")
        if self.grants_authority:
            raise ValueError("preparation scheduling factor configuration cannot grant authority")

    @property
    def phi_min(self) -> Decimal:
        return self.thresholds.phi_min

    @property
    def phi_max(self) -> Decimal:
        return self.thresholds.phi_max

    @property
    def closure_ratio_max(self) -> Decimal:
        return self.thresholds.closure_ratio_max

    @property
    def kernel_band_ratio_max(self) -> Decimal:
        return self.thresholds.kernel_band_ratio_max

    @property
    def persistence_pass_count(self) -> int:
        return self.thresholds.persistence_pass_count


def build_matrix_response_study_preparation_window_scheduling_factor_evaluation_config(
    *,
    predecessor: MatrixResponseCausalIntersectionResidenceStudyConfig,
    member: SixMatrixResponseModelFamilyMember,
    roster: SixMatrixResponseTransientControlledInvarianceProbeRoster,
) -> MatrixResponsePreparationWindowSchedulingFactorEvaluationConfig:
    """Project only the authenticated immutable causal intersection residence factor/probe choices."""

    if (
        member.member_id != predecessor.member_id
        or member.fingerprint() != predecessor.member_fingerprint
    ):
        raise ValueError("preparation scheduling factor config member differs from immutable causal intersection residence")
    expected_roster = derive_probe_roster(
        config_fingerprint=predecessor.probe_roster_config_fingerprint,
        rule_id=predecessor.probe_seed_rule_id,
        scientific_seed=int(predecessor.probe_scientific_input.scientific_seed_sha256, 16),
    )
    if (
        roster.field_ids != expected_roster.field_ids
        or roster.seed_sha256 != expected_roster.seed_sha256
        or not np.array_equal(roster.fields, expected_roster.fields)
        or not np.array_equal(roster.coordinates, expected_roster.coordinates)
    ):
        raise ValueError("preparation scheduling factor config probe roster differs from immutable transient response/causal intersection residence")
    return MatrixResponsePreparationWindowSchedulingFactorEvaluationConfig(
        config_id="matrix-preparation-scheduling.factor-evaluation.from-causal-intersection-residence",
        config_version="1.0.0",
        predecessor_factor_config=ObjectIdentity.from_record(
            predecessor.config_id, predecessor
        ),
        predecessor_factor_record=predecessor,
        family_member=ObjectIdentity.from_record(member.member_id, member),
        thresholds=MatrixResponsePreparationWindowSchedulingFactorThresholds(
            thresholds_id="matrix-preparation-scheduling.factor-thresholds.from-matrix-causal-intersection-residence",
            phi_min=predecessor.phi_min,
            phi_max=predecessor.phi_max,
            closure_ratio_max=predecessor.closure_ratio_max,
            kernel_band_ratio_max=predecessor.kernel_band_ratio_max,
            persistence_pass_count=predecessor.persistence_pass_count,
            response_geometry_loss_max=predecessor.response_geometry_loss_max,
            response_radius_ratio_max=predecessor.response_radius_ratio_max,
        ),
        q=predecessor.q,
        primary_timestep=predecessor.primary_timestep,
        receiver_cadence_steps=predecessor.receiver_cadence_steps,
        rolling_window_samples=predecessor.rolling_window_samples,
        first_eligible_post_arrival_step=256,
        last_post_arrival_step=768,
        heldout_probe_field_indices=predecessor.heldout_probe_field_indices,
        probe_kappas=predecessor.probe_kappas,
        probe_roster_field_ids=roster.field_ids,
        probe_roster_seed_sha256=roster.seed_sha256,
        probe_numeric_floor=predecessor.probe_numeric_floor,
        probe_hermiticity_residual_max=predecessor.probe_hermiticity_residual_max,
        probe_trace_residual_max=predecessor.probe_trace_residual_max,
        probe_identity_drift_max=predecessor.probe_identity_drift_max,
        probe_norm_increase_max=predecessor.probe_norm_increase_max,
        probe_forecast_lag_steps=predecessor.probe_forecast_lag_steps,
        evaluator_rule_id="matrix-preparation-scheduling.pure-factor-evaluator.from-causal-intersection-residence",
        grants_authority=False,
    )


def decode_matrix_response_study_preparation_window_scheduling_factor_evaluation_config(
    payload: bytes,
) -> MatrixResponsePreparationWindowSchedulingFactorEvaluationConfig:
    return decode_canonical_bytes(
        payload,
        MatrixResponsePreparationWindowSchedulingFactorEvaluationConfig,
        maximum_bytes=1_048_576,
    )


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise FloatingPointError("Matrix preparation scheduling factor value is nonfinite")
    return Decimal(repr(float(value)))


def _reconstruct_post_arrival_states(
    persisted: SixMatrixResponsePreparationWindowSchedulingPersistedTrajectory,
) -> tuple[SixMatrixState, ...]:
    trace = persisted.trace
    indices = np.flatnonzero(trace.post_arrival_steps >= 0)
    if not indices.size or int(trace.post_arrival_steps[indices[0]]) != 0:
        raise ValueError("persisted preparation scheduling trajectory lacks the arrival state")
    states = []
    for index in indices:
        states.append(
            SixMatrixState(
                q=2,
                positions=trace.positions[index],
                momenta=trace.momenta[index],
                step_index=int(index),
                alpha_tilde_x=float(trace.alpha_tilde[index, 0]),
                alpha_tilde_y=float(trace.alpha_tilde[index, 1]),
            )
        )
    return tuple(states)


def evaluate_persisted_trajectory_factors(
    *,
    persisted: SixMatrixResponsePreparationWindowSchedulingPersistedTrajectory,
    config: MatrixResponsePreparationWindowSchedulingFactorEvaluationConfig,
    member: SixMatrixResponseModelFamilyMember,
    roster: SixMatrixResponseTransientControlledInvarianceProbeRoster,
) -> MatrixResponsePreparationWindowSchedulingTrajectoryFactorRecord:
    """Reconstruct the complete factor ledger from decoded persisted arrays only."""

    result = persisted.result
    source_result = ObjectIdentity.from_record(result.result_id, result)
    source_measurement = ObjectIdentity.from_record(
        result.measurement_artifact.receipt_id, result.measurement_artifact
    )
    preparation_tape = ObjectIdentity.from_record(
        result.preparation_tape.tape_id, result.preparation_tape
    )
    observation_tape = ObjectIdentity.from_record(
        result.observation_tape.tape_id, result.observation_tape
    )
    factor_config = ObjectIdentity.from_record(config.config_id, config)

    def record(
        *,
        samples: tuple[MatrixResponsePreparationWindowSchedulingFactorSample, ...],
        outcome: MatrixResponsePreparationWindowSchedulingTrajectoryOutcome | None,
        disposition: MatrixResponsePreparationWindowSchedulingFactorRecordDisposition,
        reason_codes: tuple[str, ...],
    ) -> MatrixResponsePreparationWindowSchedulingTrajectoryFactorRecord:
        return MatrixResponsePreparationWindowSchedulingTrajectoryFactorRecord(
            record_id=(
                f"matrix-preparation-scheduling.factor-record.{result.tranche.value.lower()}."
                f"{result.root_block_id}.{result.schedule_id}.{result.scientific_view_id}"
            ),
            source_trajectory_result=source_result,
            source_measurement_receipt=source_measurement,
            source_development_roster=result.development_roster,
            source_scientific_view=ObjectIdentity.from_record(
                result.issued_nested_view.view_id,
                result.issued_nested_view,
            ),
            root_id=result.root_block_id,
            root_index=result.block_index,
            tranche=result.tranche,
            schedule_id=result.schedule_id,
            numerical_view_id=result.numerical_view_id,
            scientific_view_id=result.scientific_view_id,
            preparation_tape=preparation_tape,
            observation_tape=observation_tape,
            factor_config=factor_config,
            samples=samples,
            outcome=outcome,
            disposition=disposition,
            reason_codes=reason_codes,
            physical_independent_unit_id=result.physical_independent_unit_id,
            acquisition_group_id=result.acquisition_group_id,
            grants_authority=False,
        )
    if ObjectIdentity.from_record(member.member_id, member) != config.family_member:
        raise ValueError("preparation scheduling persisted factor evaluator received another family member")
    if (
        roster.field_ids != config.probe_roster_field_ids
        or roster.seed_sha256 != config.probe_roster_seed_sha256
    ):
        raise ValueError("preparation scheduling persisted factor evaluator received another probe roster")
    if (
        result.terminal is not SixMatrixResponsePreparationWindowSchedulingScheduleTerminal.COMPLETED
        or result.reason_codes
        or not result.schedule_trace.schedule_ledger.valid
    ):
        return record(
            samples=(),
            outcome=None,
            disposition=MatrixResponsePreparationWindowSchedulingFactorRecordDisposition.TECHNICAL_INVALID,
            reason_codes=result.reason_codes or ("persisted-trajectory-technical-invalid",),
        )
    multiplier = result.view_step_multiplier
    states = _reconstruct_post_arrival_states(persisted)
    expected_state_count = config.last_post_arrival_step * multiplier + 1
    if len(states) != expected_state_count:
        return record(
            samples=(),
            outcome=None,
            disposition=MatrixResponsePreparationWindowSchedulingFactorRecordDisposition.UNEVALUABLE,
            reason_codes=("persisted-post-arrival-state-sequence-incomplete",),
        )
    cadence_native = config.receiver_cadence_steps * multiplier
    receiver_states = states[::cadence_native]
    observations = tuple(
        observe_state(
            observation_id=f"matrix-preparation-scheduling.observation.{result.root_block_id}.{result.schedule_id}.{step}",
            local_step=step,
            local_time=step * float(config.primary_timestep),
            state=state,
            member=member,
            config=config,
        )
        for step, state in zip(
            range(0, config.last_post_arrival_step + 1, config.receiver_cadence_steps),
            receiver_states,
            strict=True,
        )
    )
    by_step = {value.local_step: value for value in observations}
    labels = {value.endpoint_step: value for value in rolling_labels(observations, config=config)}
    assessment_steps = tuple(
        range(
            config.first_eligible_post_arrival_step,
            config.last_post_arrival_step + 1,
            config.receiver_cadence_steps,
        )
    )
    if any(step not in by_step or step not in labels for step in assessment_steps):
        return record(
            samples=(),
            outcome=None,
            disposition=MatrixResponsePreparationWindowSchedulingFactorRecordDisposition.UNEVALUABLE,
            reason_codes=("persisted-factor-rolling-history-incomplete",),
        )
    y_path = np.ascontiguousarray(
        np.stack(tuple(value.positions[1] for value in states)), dtype="<c16"
    )
    propagation = propagate_passive_probes(
        y_path=y_path,
        start_step=0,
        timestep=float(config.primary_timestep) / multiplier,
        roster=roster,
        kappas=tuple(float(value) for value in config.probe_kappas),
        field_indices=config.heldout_probe_field_indices,
    )
    invariant_reasons: set[str] = set()
    if propagation.maximum_hermiticity_residual > float(config.probe_hermiticity_residual_max):
        invariant_reasons.add("probe-hermiticity-residual")
    if propagation.maximum_trace_residual > float(config.probe_trace_residual_max):
        invariant_reasons.add("probe-trace-residual")
    if propagation.maximum_identity_relative_drift > float(config.probe_identity_drift_max):
        invariant_reasons.add("identity-sentinel-drift")
    if propagation.maximum_relative_norm_increase > float(config.probe_norm_increase_max):
        invariant_reasons.add("probe-norm-increase")
    samples = []
    for step in assessment_steps:
        observation = by_step[step]
        label = labels[step]
        window_steps = tuple(
            range(
                step - (config.rolling_window_samples - 1) * config.receiver_cadence_steps,
                step + 1,
                config.receiver_cadence_steps,
            )
        )
        window = tuple(by_step[value] for value in window_steps)
        persistence_count = sum(value.factor_y.radius_closure_pass for value in window)
        strict_kernel = bool(
            all(value.factor_y.valid for value in window)
            and max(value.factor_y.kernel_band_ratio for value in window)
            <= config.thresholds.kernel_band_ratio_max
        )
        origin_offset = multiplier * (step - config.probe_forecast_lag_steps)
        endpoint_offset = multiplier * step
        operator = traceless_operator(y_path[origin_offset])
        radius_operator = radius_only_operator(operator)
        geometry_losses: list[float] = []
        radius_losses: list[float] = []
        resolved = 0
        for local_field in range(len(config.heldout_probe_field_indices)):
            for kappa_index, kappa in enumerate(config.probe_kappas):
                initial = propagation.states[local_field, kappa_index, origin_offset]
                observed = propagation.states[local_field, kappa_index, endpoint_offset]
                horizon = config.probe_forecast_lag_steps * float(config.primary_timestep)
                geometry_prediction = heat_predict_probe(
                    operator=operator,
                    probe=initial,
                    kappa=float(kappa),
                    horizon_time=horizon,
                )
                radius_prediction = heat_predict_probe(
                    operator=radius_operator,
                    probe=initial,
                    kappa=float(kappa),
                    horizon_time=horizon,
                )
                geometry_loss, is_resolved = normalized_increment_loss(
                    predicted=geometry_prediction,
                    observed=observed,
                    initial=initial,
                    floor=float(config.probe_numeric_floor),
                )
                radius_loss, _ = normalized_increment_loss(
                    predicted=radius_prediction,
                    observed=observed,
                    initial=initial,
                    floor=float(config.probe_numeric_floor),
                )
                if is_resolved:
                    resolved += 1
                    geometry_losses.append(geometry_loss)
                    radius_losses.append(radius_loss)
        geometry_mean = float(np.mean(geometry_losses)) if geometry_losses else None
        radius_mean = float(np.mean(radius_losses)) if radius_losses else None
        ratio = (
            geometry_mean / radius_mean
            if geometry_mean is not None
            and radius_mean is not None
            and radius_mean > np.finfo(np.float64).tiny
            else None
        )
        operands = MatrixResponsePreparationWindowSchedulingFactorOperands(
            operand_id=(
                f"matrix-preparation-scheduling.operand.{result.tranche.value.lower()}.{result.root_block_id}."
                f"{result.schedule_id}.{result.scientific_view_id}.step-{step}"
            ),
            root_id=result.root_block_id,
            root_index=result.block_index,
            tranche=result.tranche,
            schedule_id=result.schedule_id,
            numerical_view_id=result.numerical_view_id,
            scientific_view_id=result.scientific_view_id,
            acquisition_group_id=result.acquisition_group_id,
            physical_independent_unit_id=result.physical_independent_unit_id,
            parent_step=step,
            post_arrival_time=Decimal(step) * config.primary_timestep,
            thresholds=config.thresholds,
            phi_y=observation.factor_y.phi,
            closure_y=observation.factor_y.closure_ratio,
            kernel_band_y=observation.factor_y.kernel_band_ratio,
            spectrum_valid=observation.factor_y.valid,
            persistence_pass_count=persistence_count,
            strict_kernel_window_pass=strict_kernel,
            x_rolling_geometric=label.x_geometric,
            resolved_probe_cells=resolved,
            required_probe_cells=18,
            probe_invariants_pass=not invariant_reasons,
            geometry_loss=None if geometry_mean is None else _decimal(geometry_mean),
            radius_loss=None if radius_mean is None else _decimal(radius_mean),
            geometry_radius_ratio=None if ratio is None else _decimal(ratio),
        )
        samples.append(evaluate_factor_sample(operands))
    ordered_samples = tuple(samples)
    outcome = reduce_trajectory_window(ordered_samples)
    return record(
        samples=ordered_samples,
        outcome=outcome,
        disposition=MatrixResponsePreparationWindowSchedulingFactorRecordDisposition.EVALUATED,
        reason_codes=(),
    )
