"""observation order geometry projection, interval inference, and two-view adjudication."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.simulators.six_matrix_response.model import SixMatrixState
from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView
from empirical_lawhood.adapters.simulators.six_matrix_response.observation_order import MatrixObservationOrderAcquisitionDisposition, MatrixObservationOrderPairedHistoryResult, read_observation_order_paired_history
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import derive_probe_roster, heat_predict_probe, normalized_increment_loss, propagate_passive_probes, radius_only_operator, traceless_operator
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from empirical_lawhood.adapters.simulators.six_matrix_response.history_preparation import FOUR_FAMILY_PREPARATION_IDS

from .history_conditioned_geometry import ClusteredProbabilityInterval, EventInterval, GeometryEventKind, GeometryObservationStatus, GeometryState, GeometryTick, HistoryViewTrace, IntervalDisposition, ScientificView, TwoViewTerminal, ViewDecision, ViewQualitativeDecision, adjudicate_two_views, derive_history_events
from .shooting_committor import observe_state, rolling_labels


OBSERVATION_ORDER_PROJECTION_MAXIMUM_BYTES = 512 * 1024
OBSERVATION_ORDER_EVALUATION_MAXIMUM_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderProjectionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-observation-order-projection-config'

    config_id: str
    science_specification: ObjectIdentity
    source_config: ObjectIdentity
    member: SixMatrixResponseModelFamilyMember
    primary_view: SixMatrixResponseNumericalView
    fine_view: SixMatrixResponseNumericalView
    family_ids: tuple[str, ...]
    receiver_lattice: tuple[int, ...]
    branch_id: str
    receiver_cadence_steps: int
    rolling_window_samples: int
    persistence_pass_count: int
    probe_forecast_lag_steps: int
    probe_field_indices: tuple[int, ...]
    probe_kappas: tuple[Decimal, ...]
    probe_seed_sha256: str
    probe_roster_scientific_seed_sha256: str
    phi_min: Decimal
    phi_max: Decimal
    closure_ratio_max: Decimal
    kernel_band_ratio_max: Decimal
    probe_numeric_floor: Decimal
    probe_hermiticity_residual_max: Decimal
    probe_trace_residual_max: Decimal
    probe_identity_drift_max: Decimal
    probe_norm_increase_max: Decimal
    probe_geometry_loss_max: Decimal
    probe_geometry_radius_ratio_max: Decimal
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    q: int = 2

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.branch_id, field_name="branch_id")
        if self.family_ids != FOUR_FAMILY_PREPARATION_IDS:
            raise ValueError("observation-order family roster differs from its fixed scientific order")
        if self.receiver_lattice != tuple(range(0, 1025, 16)):
            raise ValueError("observation order projection receiver lattice differs")
        if self.probe_field_indices != tuple(range(12)):
            raise ValueError("observation order projection must use all twelve frozen probes")
        validate_sha256(self.probe_seed_sha256, field_name="probe_seed_sha256")
        validate_sha256(self.probe_roster_scientific_seed_sha256, field_name="probe_roster_scientific_seed_sha256")
        if (
            self.probe_seed_sha256 != 'dd996dd7b74e2ce62b2ec8a1c174a7a73b0bc18ac56048968f345f6fcdb16cc7'
            or self.probe_roster_scientific_seed_sha256 != '6eb889bd104fb772b64ade4d755d7c76e02094faee4053b3f9b29cde279b99da'
        ):
            raise ValueError("observation-order probe scientific seed differs from its fixed input")
        for name in (
            "phi_min",
            "phi_max",
            "closure_ratio_max",
            "kernel_band_ratio_max",
            "probe_numeric_floor",
            "probe_hermiticity_residual_max",
            "probe_trace_residual_max",
            "probe_identity_drift_max",
            "probe_norm_increase_max",
            "probe_geometry_loss_max",
            "probe_geometry_radius_ratio_max",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if (
            self.receiver_cadence_steps != 16
            or self.member.member_id != "six-matrix-response.member.mass-0p5.cross-coupling-1"
            or self.member.fingerprint()
            != "1f15e5757d5f1adef0bb6611d83a6210dd5e1b2d94b5a1adef1639124192c287"
            or self.primary_view.view_id != "six-matrix-response.view.baoab-dt-0p001"
            or self.primary_view.fingerprint()
            != "8abd68021c070d1fe82e32c6a0b488b2e559f213be000c5873a18742eafb4d6a"
            or self.fine_view.view_id != "six-matrix-response.view.baoab-dt-0p0005"
            or self.fine_view.fingerprint()
            != "bf595a377ad9a4f9a064b1640bee26b864f1bc8e2256a88ca8cf4e6fa55bce99"
            or self.rolling_window_samples != 16
            or self.persistence_pass_count != 12
            or self.probe_forecast_lag_steps != 32
            or self.probe_kappas != (Decimal("0.25"), Decimal("0.5"), Decimal("1.0"))
            or self.phi_min != Decimal("0.35")
            or self.phi_max != Decimal("0.95")
            or self.closure_ratio_max != Decimal("0.30")
            or self.kernel_band_ratio_max != Decimal("0.25")
            or self.probe_numeric_floor != Decimal("1e-12")
            or self.probe_hermiticity_residual_max != Decimal("1e-10")
            or self.probe_trace_residual_max != Decimal("1e-10")
            or self.probe_identity_drift_max != Decimal("1e-10")
            or self.probe_norm_increase_max != Decimal("1e-10")
            or self.probe_geometry_loss_max != Decimal("0.25")
            or self.probe_geometry_radius_ratio_max != Decimal("0.75")
            or self.q != 2
            or self.grants_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("observation order projection config differs from its frozen contract")


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderEvaluationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-observation-order-evaluation-config'

    config_id: str
    science_specification: ObjectIdentity
    projection_config: ObjectIdentity
    family_ids: tuple[str, ...]
    cutoff_ticks: tuple[int, ...]
    horizon_offsets: tuple[int, ...]
    transition_lags: tuple[int, ...]
    bootstrap_seed_sha256: str
    bootstrap_replicates: int
    confidence_level: Decimal
    minimum_defined_replicates: int
    minimum_known_risk_overall: int
    minimum_known_risk_per_family: int
    transition_minimum_unique_histories: int
    transition_minimum_available_pairs: int
    retention_anchor_horizon: int
    reentry_anchor_horizon: int
    retention_lower_bound: Decimal
    reentry_lower_bound: Decimal
    expected_physical_histories: int
    expected_traces: int
    qualification_fixture_id: str | None
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.qualification_fixture_id is not None:
            validate_stable_id(
                self.qualification_fixture_id,
                field_name="qualification_fixture_id",
            )
        if self.family_ids != FOUR_FAMILY_PREPARATION_IDS:
            raise ValueError("observation-order family roster differs from its fixed scientific order")
        validate_sha256(self.bootstrap_seed_sha256, field_name="bootstrap_seed_sha256")
        for name in (
            "confidence_level",
            "retention_lower_bound",
            "reentry_lower_bound",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if (
            self.cutoff_ticks != (384, 512, 640, 768)
            or self.horizon_offsets != (64, 128, 256)
            or self.transition_lags != (16, 32, 64, 128)
            or self.bootstrap_replicates != 4096
            or self.confidence_level != Decimal("0.95")
            or self.minimum_defined_replicates != 3687
            or self.minimum_known_risk_overall != 16
            or self.minimum_known_risk_per_family != 4
            or self.transition_minimum_unique_histories != 4
            or self.transition_minimum_available_pairs != 16
            or self.retention_anchor_horizon != 64
            or self.reentry_anchor_horizon != 256
            or self.retention_lower_bound != Decimal("0.50")
            or self.reentry_lower_bound != Decimal("0.10")
            or (self.expected_physical_histories, self.expected_traces)
            not in {(256, 512), (4, 8)}
            or ((self.expected_physical_histories, self.expected_traces) == (256, 512))
            != (self.qualification_fixture_id is None)
            or self.grants_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("observation order evaluation config differs from its frozen contract")


def _decimal(value: float) -> Decimal:
    return Decimal(repr(float(value)))


def _unavailable_trace(
    result: MatrixObservationOrderPairedHistoryResult,
    config: MatrixObservationOrderProjectionConfig,
    view: ScientificView,
) -> HistoryViewTrace:
    reason = (
        "NATIVE_NUMERICAL_INVALID"
        if result.disposition is MatrixObservationOrderAcquisitionDisposition.INVALID
        else "NATIVE_HISTORY_PARTIAL"
        if result.disposition is MatrixObservationOrderAcquisitionDisposition.PARTIAL
        else "NATIVE_PROVIDER_EXCEPTION"
    )
    ticks = tuple(
        GeometryTick(
            tick_id=f"tick.{result.history_id}.{view.value.lower()}.{step:04d}",
            history_id=result.history_id,
            preparation_family_id=result.family_id,
            acquisition_group_id=result.acquisition_group_id,
            scientific_view=view,
            receiver_tick=step,
            receiver_time=Decimal(step) * Decimal("0.001"),
            time_unit="dimensionless-langevin-time",
            frame_id="six-matrix-response.frame.simultaneous-unitary-quotient",
            receiver_direction="forward-autonomous-history",
            required_window_start_tick=max(0, step - 240),
            status=GeometryObservationStatus.INVALID,
            full_intersection_pass=None,
            robust_00_pass=None,
            reason_codes=(reason,),
        )
        for step in config.receiver_lattice
    )
    return HistoryViewTrace(
        trace_id=f"trace.{result.history_id}.{view.value.lower()}",
        history_id=result.history_id,
        preparation_family_id=result.family_id,
        acquisition_group_id=result.acquisition_group_id,
        scientific_view=view,
        branch_id=config.branch_id,
        receiver_lattice=config.receiver_lattice,
        ticks=ticks,
    )


def project_observation_order_history_view(
    *,
    payload: bytes,
    result: MatrixObservationOrderPairedHistoryResult,
    config: MatrixObservationOrderProjectionConfig,
    view: ScientificView,
) -> HistoryViewTrace:
    """Project one nested view without changing acquisition or physical n."""

    if (
        result.family_id not in config.family_ids
        or config.source_config != result.source_config
    ):
        raise ValueError("observation order projection source/config lineage differs")
    if result.disposition is not MatrixObservationOrderAcquisitionDisposition.COMPLETE:
        return _unavailable_trace(result, config, view)
    view_name = "primary" if view is ScientificView.PRIMARY else "fine"
    expected_view_id = (
        result.primary_view_id
        if view is ScientificView.PRIMARY
        else result.fine_view_id
    )
    if expected_view_id not in {result.primary_view_id, result.fine_view_id}:
        raise ValueError("observation order projection nested-view identity differs")
    positions, momenta, alpha = (
        read_observation_order_paired_history(payload, result, "primary")
        if view_name == "primary"
        else read_observation_order_paired_history(payload, result, "fine")
    )
    multiplier = 1 if view is ScientificView.PRIMARY else 2
    numerical = config.primary_view if multiplier == 1 else config.fine_view
    observations = []
    states: dict[int, SixMatrixState] = {}
    for step in config.receiver_lattice:
        native = step * multiplier
        state = SixMatrixState(
            q=2,
            positions=positions[native],
            momenta=momenta[native],
            step_index=native,
            alpha_tilde_x=float(alpha[native, 0]),
            alpha_tilde_y=float(alpha[native, 1]),
        )
        states[step] = state
        observations.append(
            observe_state(
                observation_id=f"observation.{result.history_id}.{view.value.lower()}.{step:04d}",
                local_step=step,
                local_time=step * float(config.primary_view.timestep),
                state=state,
                member=config.member,
                config=config,
            )
        )
    labels = {
        value.endpoint_step: value
        for value in rolling_labels(tuple(observations), config=config)
    }
    by_step = {value.local_step: value for value in observations}
    roster = derive_probe_roster(
        config_fingerprint=config.probe_seed_sha256,
        rule_id="matrix-observation-order.geometry-probe-roster",
        scientific_seed=int(config.probe_roster_scientific_seed_sha256, 16),
    )
    propagation = propagate_passive_probes(
        y_path=np.asarray(positions[:, 1], dtype="<c16"),
        start_step=0,
        timestep=float(numerical.timestep),
        roster=roster,
        field_indices=config.probe_field_indices,
    )
    invariant_reasons = tuple(
        reason
        for reason, failed in (
            (
                "PROBE_HERMITICITY_INVALID",
                propagation.maximum_hermiticity_residual
                > float(config.probe_hermiticity_residual_max),
            ),
            (
                "PROBE_TRACE_INVALID",
                propagation.maximum_trace_residual
                > float(config.probe_trace_residual_max),
            ),
            (
                "PROBE_IDENTITY_DRIFT",
                propagation.maximum_identity_relative_drift
                > float(config.probe_identity_drift_max),
            ),
            (
                "PROBE_NORM_INCREASE",
                propagation.maximum_relative_norm_increase
                > float(config.probe_norm_increase_max),
            ),
        )
        if failed
    )
    ticks = []
    for step in config.receiver_lattice:
        start = max(0, step - 240)
        if step < 240:
            ticks.append(
                GeometryTick(
                    f"tick.{result.history_id}.{view.value.lower()}.{step:04d}",
                    result.history_id,
                    result.family_id,
                    result.acquisition_group_id,
                    view,
                    step,
                    Decimal(step) * Decimal("0.001"),
                    "dimensionless-langevin-time",
                    "six-matrix-response.frame.simultaneous-unitary-quotient",
                    "forward-autonomous-history",
                    start,
                    GeometryObservationStatus.UNAVAILABLE,
                    None,
                    None,
                    ("ROLLING_WINDOW_INCOMPLETE",),
                )
            )
            continue
        label = labels.get(step)
        if label is None or invariant_reasons:
            ticks.append(
                GeometryTick(
                    f"tick.{result.history_id}.{view.value.lower()}.{step:04d}",
                    result.history_id,
                    result.family_id,
                    result.acquisition_group_id,
                    view,
                    step,
                    Decimal(step) * Decimal("0.001"),
                    "dimensionless-langevin-time",
                    "six-matrix-response.frame.simultaneous-unitary-quotient",
                    "forward-autonomous-history",
                    start,
                    GeometryObservationStatus.INVALID,
                    None,
                    None,
                    invariant_reasons or ("ROLLING_NORMALIZATION_INVALID",),
                )
            )
            continue
        observation = by_step[step]
        window = tuple(by_step[value] for value in range(start, step + 1, 16))
        origin_native = (step - config.probe_forecast_lag_steps) * multiplier
        endpoint_native = step * multiplier
        operator = traceless_operator(positions[origin_native, 1])
        radius_operator = radius_only_operator(operator)
        geometry_losses: list[float] = []
        radius_losses: list[float] = []
        for field_index in range(12):
            for kappa_index, kappa in enumerate(config.probe_kappas):
                initial = propagation.states[field_index, kappa_index, origin_native]
                observed = propagation.states[field_index, kappa_index, endpoint_native]
                horizon = config.probe_forecast_lag_steps * float(
                    config.primary_view.timestep
                )
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
                geometry_loss, resolved = normalized_increment_loss(
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
                if resolved:
                    geometry_losses.append(geometry_loss)
                    radius_losses.append(radius_loss)
        geometry = (
            float(np.mean(geometry_losses)) if len(geometry_losses) == 36 else None
        )
        radius = float(np.mean(radius_losses)) if len(radius_losses) == 36 else None
        ratio = (
            geometry / radius
            if geometry is not None and radius is not None and radius > 1e-300
            else None
        )
        probe_pass = bool(
            geometry is not None
            and ratio is not None
            and geometry <= float(config.probe_geometry_loss_max)
            and ratio <= float(config.probe_geometry_radius_ratio_max)
        )
        y = observation.factor_y
        y_structural = bool(
            float(config.phi_min) <= float(y.phi) <= float(config.phi_max)
            and float(y.closure_ratio) <= float(config.closure_ratio_max)
            and y.valid
            and float(y.kernel_band_ratio) <= float(config.kernel_band_ratio_max)
            and sum(value.factor_y.radius_closure_pass for value in window)
            >= config.persistence_pass_count
            and max(float(value.factor_y.kernel_band_ratio) for value in window)
            <= float(config.kernel_band_ratio_max)
        )
        full = probe_pass and y_structural and not label.x_geometric
        robust = not label.x_geometric and not label.y_geometric
        ticks.append(
            GeometryTick(
                f"tick.{result.history_id}.{view.value.lower()}.{step:04d}",
                result.history_id,
                result.family_id,
                result.acquisition_group_id,
                view,
                step,
                Decimal(step) * Decimal("0.001"),
                "dimensionless-langevin-time",
                "six-matrix-response.frame.simultaneous-unitary-quotient",
                "forward-autonomous-history",
                start,
                GeometryObservationStatus.AVAILABLE,
                full,
                robust,
                (),
            )
        )
    return HistoryViewTrace(
        trace_id=f"trace.{result.history_id}.{view.value.lower()}",
        history_id=result.history_id,
        preparation_family_id=result.family_id,
        acquisition_group_id=result.acquisition_group_id,
        scientific_view=view,
        branch_id=config.branch_id,
        receiver_lattice=config.receiver_lattice,
        ticks=tuple(ticks),
    )


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderNonparametricMaximumLikelihoodPoint(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-observation-order-nonparametric-maximum-likelihood-point'

    receiver_tick: int
    at_risk_weight: Decimal
    event_weight: Decimal
    hazard: Decimal | None
    survival: Decimal | None


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderIntervalEstimate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-observation-order-interval-estimate'

    estimate_id: str
    scientific_view: ScientificView
    preparation_family_id: str
    cutoff_tick: int
    horizon_tick: int
    event_kind: GeometryEventKind
    physical_history_count: int
    known_risk_count: int
    event_count: int
    right_censored_count: int
    unknown_count: int
    not_at_risk_count: int
    event_intervals: tuple[EventInterval, ...]
    curve: tuple[MatrixObservationOrderNonparametricMaximumLikelihoodPoint, ...]
    endpoint_estimate: Decimal | None
    support_sufficient: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.estimate_id, field_name="estimate_id")
        validate_stable_id(
            self.preparation_family_id, field_name="preparation_family_id"
        )
        if (
            sum(
                (
                    self.event_count,
                    self.right_censored_count,
                    self.unknown_count,
                    self.not_at_risk_count,
                )
            )
            != self.physical_history_count
        ):
            raise ValueError("observation order interval disposition counts differ from physical n")
        if self.known_risk_count != self.event_count + self.right_censored_count:
            raise ValueError("observation order interval risk denominator differs")
        if len(self.event_intervals) != self.physical_history_count:
            raise ValueError("observation order interval roster differs from physical n")


def _npmle(
    *,
    intervals: tuple[EventInterval, ...],
    family: str,
    view: ScientificView,
    minimum: int,
) -> MatrixObservationOrderIntervalEstimate:
    first = intervals[0]
    known = tuple(
        value
        for value in intervals
        if value.disposition
        in {IntervalDisposition.EVENT, IntervalDisposition.RIGHT_CENSORED}
    )
    survival = 1.0
    curve: list[MatrixObservationOrderNonparametricMaximumLikelihoodPoint] = []
    lattice = range(first.cutoff_tick + 16, first.horizon_tick + 1, 16)
    for tick in lattice:
        risk = sum(
            (
                value.upper_tick is not None and value.upper_tick >= tick
                if value.disposition is IntervalDisposition.EVENT
                else value.lower_tick is not None and value.lower_tick >= tick
            )
            for value in known
        )
        events = sum(
            value.disposition is IntervalDisposition.EVENT
            and value.upper_tick == tick
            for value in known
        )
        hazard = events / risk if risk else None
        if hazard is not None:
            survival *= 1.0 - hazard
        curve.append(
            MatrixObservationOrderNonparametricMaximumLikelihoodPoint(
                receiver_tick=tick,
                at_risk_weight=Decimal(risk),
                event_weight=Decimal(events),
                hazard=None if hazard is None else _decimal(hazard),
                survival=None if hazard is None and not curve else _decimal(survival),
            )
        )
    endpoint = (
        None
        if not known
        else _decimal(
            survival if first.event_kind is GeometryEventKind.EXIT else 1.0 - survival
        )
    )
    counts = {
        value: sum(item.disposition is value for item in intervals)
        for value in IntervalDisposition
    }
    return MatrixObservationOrderIntervalEstimate(
        estimate_id=(
            f"npmle.{view.value.lower()}.{family}.{first.event_kind.value.lower()}."
            f"c{first.cutoff_tick}.h{first.horizon_tick}"
        ),
        scientific_view=view,
        preparation_family_id=family,
        cutoff_tick=first.cutoff_tick,
        horizon_tick=first.horizon_tick,
        event_kind=first.event_kind,
        physical_history_count=len(intervals),
        known_risk_count=len(known),
        event_count=counts[IntervalDisposition.EVENT],
        right_censored_count=counts[IntervalDisposition.RIGHT_CENSORED],
        unknown_count=counts[IntervalDisposition.UNKNOWN],
        not_at_risk_count=counts[IntervalDisposition.NOT_AT_RISK],
        event_intervals=intervals,
        curve=tuple(curve),
        endpoint_estimate=endpoint,
        support_sufficient=len(known) >= minimum,
    )


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderTransitionCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-observation-order-transition-cell'

    from_state: GeometryState
    to_state: GeometryState
    opportunity_count: int
    unique_history_count: int
    probability: Decimal | None
    lower: Decimal | None
    upper: Decimal | None
    support_sufficient: bool


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderTransitionMatrix(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-observation-order-transition-matrix'

    matrix_id: str
    scientific_view: ScientificView
    preparation_family_id: str
    receiver_lag: int
    physical_history_count: int
    cells: tuple[MatrixObservationOrderTransitionCell, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.matrix_id, field_name="matrix_id")
        validate_stable_id(
            self.preparation_family_id, field_name="preparation_family_id"
        )
        expected = tuple(
            (left, right) for left in GeometryState for right in GeometryState
        )
        if (
            tuple((value.from_state, value.to_state) for value in self.cells)
            != expected
        ):
            raise ValueError("observation order transition matrix cell order differs")


def _original_transition_family_entropy_coordinate(family: str) -> str:
    """Return the original scientific KDF coordinate for a current family.

    These four original source coordinates affect bootstrap draws. They are
    used only at the hash boundary and are not accepted current family IDs.
    """

    if type(family) is not str or family not in FOUR_FAMILY_PREPARATION_IDS:
        raise ValueError("transition bootstrap requires an exact current preparation family")
    original_coordinates = (
        "cc1.history.co-anneal",
        "cc1.history.de-anneal",
        "cc1.history.x-first",
        "cc1.history.y-first",
    )
    return original_coordinates[FOUR_FAMILY_PREPARATION_IDS.index(family)]


def _transition_matrix(
    traces: tuple[HistoryViewTrace, ...],
    config: MatrixObservationOrderEvaluationConfig,
    family: str,
    view: ScientificView,
    lag: int,
) -> MatrixObservationOrderTransitionMatrix:
    selected = tuple(
        value
        for value in traces
        if value.preparation_family_id == family and value.scientific_view is view
    )
    counts = np.zeros((len(selected), 4, 4), dtype=np.int64)
    for index, trace in enumerate(selected):
        by_tick = {value.receiver_tick: value for value in trace.ticks}
        for left_tick in trace.ticks:
            right_tick = by_tick.get(left_tick.receiver_tick + lag)
            if (
                left_tick.geometry_state is not None
                and right_tick is not None
                and right_tick.geometry_state is not None
            ):
                counts[
                    index,
                    list(GeometryState).index(left_tick.geometry_state),
                    list(GeometryState).index(right_tick.geometry_state),
                ] += 1
    total = counts.sum(axis=0)
    rng_seed = sha256(
        f"{config.bootstrap_seed_sha256}\0transition\0{_original_transition_family_entropy_coordinate(family)}\0{view.value}\0{lag}".encode()
    ).digest()
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(rng_seed[:16], "big")))
    draws = rng.integers(
        0, len(selected), size=(config.bootstrap_replicates, len(selected))
    )
    sampled = counts[draws].sum(axis=1)
    cells = []
    for left_index, left_state in enumerate(GeometryState):
        denominator = int(total[left_index].sum())
        unique = int(np.count_nonzero(counts[:, left_index].sum(axis=1)))
        support = (
            unique >= config.transition_minimum_unique_histories
            and denominator >= config.transition_minimum_available_pairs
        )
        sampled_denominator = sampled[:, left_index].sum(axis=1)
        for right_index, right_state in enumerate(GeometryState):
            probability = (
                total[left_index, right_index] / denominator if denominator else None
            )
            valid = sampled_denominator > 0
            probabilities = (
                sampled[valid, left_index, right_index] / sampled_denominator[valid]
            )
            lower = upper = None
            if support and len(probabilities) >= config.minimum_defined_replicates:
                lower, upper = np.quantile(
                    probabilities, (0.025, 0.975), method="linear"
                )
            else:
                support = False
            cells.append(
                MatrixObservationOrderTransitionCell(
                    left_state,
                    right_state,
                    denominator,
                    unique,
                    None if probability is None else _decimal(probability),
                    None if lower is None else _decimal(float(lower)),
                    None if upper is None else _decimal(float(upper)),
                    support,
                )
            )
    return MatrixObservationOrderTransitionMatrix(
        f"transition.{view.value.lower()}.{family}.lag-{lag}",
        view,
        family,
        lag,
        len(selected),
        tuple(cells),
    )


def _history_metrics(
    traces: tuple[HistoryViewTrace, ...],
    config: MatrixObservationOrderEvaluationConfig,
    view: ScientificView,
) -> tuple[dict[str, float], dict[str, float]]:
    retention: dict[str, float] = {}
    reentry: dict[str, float] = {}
    for trace in (value for value in traces if value.scientific_view is view):
        retained: list[float] = []
        reentered: list[float] = []
        for cutoff in config.cutoff_ticks:
            retention_events = derive_history_events(
                trace,
                cutoff_tick=cutoff,
                horizon_tick=cutoff + config.retention_anchor_horizon,
            )
            if retention_events.exit.disposition in {
                IntervalDisposition.EVENT,
                IntervalDisposition.RIGHT_CENSORED,
            }:
                retained.append(
                    float(retention_events.retained_through_horizon is True)
                )
            reentry_events = derive_history_events(
                trace,
                cutoff_tick=cutoff,
                horizon_tick=cutoff + config.reentry_anchor_horizon,
            )
            value = reentry_events.history_qualified_reentry
            if value.disposition in {
                IntervalDisposition.EVENT,
                IntervalDisposition.RIGHT_CENSORED,
            }:
                reentered.append(
                    float(value.disposition is IntervalDisposition.EVENT)
                )
        if retained:
            retention[trace.history_id] = float(np.mean(retained))
        if reentered:
            reentry[trace.history_id] = float(np.mean(reentered))
    return retention, reentry


def _paired_intervals(
    traces: tuple[HistoryViewTrace, ...], config: MatrixObservationOrderEvaluationConfig
) -> dict[tuple[ScientificView, str], ClusteredProbabilityInterval]:
    by_history = {value.history_id: value.preparation_family_id for value in traces}
    if len(by_history) != config.expected_physical_histories:
        raise ValueError("observation order inference physical n differs")
    metrics = {
        view: _history_metrics(traces, config, view) for view in ScientificView
    }
    roster = {
        family: tuple(
            sorted(
                history
                for history, observed in by_history.items()
                if observed == family
            )
        )
        for family in config.family_ids
    }
    rng = np.random.Generator(
        np.random.PCG64DXSM(
            int.from_bytes(bytes.fromhex(config.bootstrap_seed_sha256)[:16], "big")
        )
    )
    draws = {
        family: rng.integers(
            0,
            len(roster[family]),
            size=(config.bootstrap_replicates, len(roster[family])),
        )
        for family in config.family_ids
    }
    output = {}
    for view in ScientificView:
        for metric_index, metric_name in enumerate(("retention", "reentry")):
            values = metrics[view][metric_index]
            known_by_family = {
                family: tuple(
                    history for history in roster[family] if history in values
                )
                for family in config.family_ids
            }
            known_count = sum(len(value) for value in known_by_family.values())
            initial_support = known_count >= config.minimum_known_risk_overall and all(
                len(known_by_family[family]) >= config.minimum_known_risk_per_family
                for family in config.family_ids
            )
            point = None
            if initial_support:
                point = float(
                    np.mean(
                        [
                            np.mean(
                                [values[history] for history in known_by_family[family]]
                            )
                            for family in config.family_ids
                        ]
                    )
                )
            samples = []
            if initial_support:
                for replicate in range(config.bootstrap_replicates):
                    family_values = []
                    for family in config.family_ids:
                        sampled = [
                            roster[family][index]
                            for index in draws[family][replicate]
                            if roster[family][index] in values
                        ]
                        if not sampled:
                            break
                        family_values.append(
                            float(np.mean([values[history] for history in sampled]))
                        )
                    if len(family_values) == len(config.family_ids):
                        samples.append(float(np.mean(family_values)))
            support = (
                initial_support and len(samples) >= config.minimum_defined_replicates
            )
            lower = upper = None
            if support:
                lower, upper = np.quantile(samples, (0.025, 0.975), method="linear")
            output[(view, metric_name)] = ClusteredProbabilityInterval(
                physical_history_count=len(by_history),
                known_risk_count=known_count,
                estimate=None if not support else _decimal(point or 0.0),
                lower=None if lower is None else _decimal(float(lower)),
                upper=None if upper is None else _decimal(float(upper)),
                confidence_level=config.confidence_level,
                defined_replicates=len(samples),
                undefined_replicates=config.bootstrap_replicates - len(samples),
                support_sufficient=support,
            )
    return output


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderViewReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-observation-order-view-report'

    report_id: str
    scientific_view: ScientificView
    physical_history_count: int
    family_history_counts: tuple[tuple[str, int], ...]
    available_tick_count: int
    unavailable_tick_count: int
    invalid_tick_count: int
    interval_estimates: tuple[MatrixObservationOrderIntervalEstimate, ...]
    transition_matrices: tuple[MatrixObservationOrderTransitionMatrix, ...]
    retention: ClusteredProbabilityInterval
    history_qualified_reentry: ClusteredProbabilityInterval
    decision: ViewQualitativeDecision

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        if (
            sum(value[1] for value in self.family_history_counts)
            != self.physical_history_count
        ):
            raise ValueError("observation order view family denominators differ from physical n")


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-observation-order-evaluation'

    evaluation_id: str
    analysis_identity: str
    primary_report: ObjectIdentity
    fine_report: ObjectIdentity
    physical_history_count: int
    nested_view_count: int
    qualification_fixture_id: str | None
    terminal: TwoViewTerminal
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        validate_stable_id(self.analysis_identity, field_name="analysis_identity")
        require_sorted_unique_strings(
            self.reason_codes, field_name="reason_codes", allow_empty=False
        )
        if (
            (self.physical_history_count, self.nested_view_count)
            not in {(256, 512), (4, 8)}
            or ((self.physical_history_count, self.nested_view_count) == (256, 512))
            != (self.qualification_fixture_id is None)
            or self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("observation order evaluation denominator or visibility differs")


def evaluate_observation_order(
    *,
    traces: tuple[HistoryViewTrace, ...],
    config: MatrixObservationOrderEvaluationConfig,
    analysis_identity: str,
) -> tuple[MatrixObservationOrderViewReport, MatrixObservationOrderViewReport, MatrixObservationOrderEvaluation]:
    """Evaluate the complete paired roster without cross-view compensation."""

    if len(traces) != config.expected_traces:
        raise ValueError("observation order evaluator lacks the complete paired trace roster")
    keys = {(value.history_id, value.scientific_view) for value in traces}
    histories = {value.history_id: value for value in traces}
    if len(keys) != len(traces) or len(histories) != config.expected_physical_histories:
        raise ValueError("observation order evaluator received duplicate or missing nested views")
    for history_id in histories:
        paired = tuple(value for value in traces if value.history_id == history_id)
        if (
            {value.scientific_view for value in paired} != set(ScientificView)
            or len({value.acquisition_group_id for value in paired}) != 1
            or len({value.preparation_family_id for value in paired}) != 1
        ):
            raise ValueError("observation order evaluator received a cross-history view pair")
    intervals = _paired_intervals(traces, config)
    reports = []
    for view in ScientificView:
        selected = tuple(value for value in traces if value.scientific_view is view)
        estimates = []
        for family in config.family_ids:
            family_traces = tuple(
                value for value in selected if value.preparation_family_id == family
            )
            for cutoff in config.cutoff_ticks:
                for offset in config.horizon_offsets:
                    event_sets = tuple(
                        derive_history_events(
                            trace,
                            cutoff_tick=cutoff,
                            horizon_tick=cutoff + offset,
                        )
                        for trace in family_traces
                    )
                    for kind, attribute in (
                        (GeometryEventKind.ENTRY, "entry"),
                        (GeometryEventKind.EXIT, "exit"),
                        (
                            GeometryEventKind.HISTORY_QUALIFIED_REENTRY,
                            "history_qualified_reentry",
                        ),
                        (
                            GeometryEventKind.POST_CUTOFF_EXIT_REENTRY,
                            "post_cutoff_exit_reentry",
                        ),
                    ):
                        estimates.append(
                            _npmle(
                                intervals=tuple(
                                    getattr(value, attribute) for value in event_sets
                                ),
                                family=family,
                                view=view,
                                minimum=config.minimum_known_risk_per_family,
                            )
                        )
        transitions = tuple(
            _transition_matrix(traces, config, family, view, lag)
            for family in config.family_ids
            for lag in config.transition_lags
        )
        retention = intervals[(view, "retention")]
        reentry = intervals[(view, "reentry")]
        operational = not any(
            "NATIVE_PROVIDER_EXCEPTION" in tick.reason_codes
            or "NATIVE_HISTORY_PARTIAL" in tick.reason_codes
            for trace in selected
            for tick in trace.ticks
        )
        numerical = not any(
            "NATIVE_NUMERICAL_INVALID" in tick.reason_codes
            for trace in selected
            for tick in trace.ticks
        )
        support = retention.support_sufficient and reentry.support_sufficient
        retention_pass = bool(
            support
            and retention.lower is not None
            and retention.lower >= config.retention_lower_bound
        )
        reentry_pass = bool(
            support
            and reentry.lower is not None
            and reentry.lower >= config.reentry_lower_bound
        )
        expected = (
            ViewDecision.OPERATIONALLY_UNEVALUABLE
            if not operational
            else ViewDecision.NUMERICAL_UNEVALUABLE
            if not numerical
            else ViewDecision.STRATUM_UNEVALUABLE
            if not support
            else ViewDecision.QUALITATIVE_PASS
            if retention_pass and reentry_pass
            else ViewDecision.NOT_CONFIRMED
        )
        counts = {
            status: sum(
                tick.status is status for trace in selected for tick in trace.ticks
            )
            for status in GeometryObservationStatus
        }
        reports.append(
            MatrixObservationOrderViewReport(
                report_id=f"view-report.{analysis_identity}.{view.value.lower()}",
                scientific_view=view,
                physical_history_count=len({value.history_id for value in selected}),
                family_history_counts=tuple(
                    (
                        family,
                        len(
                            {
                                value.history_id
                                for value in selected
                                if value.preparation_family_id == family
                            }
                        ),
                    )
                    for family in config.family_ids
                ),
                available_tick_count=counts[GeometryObservationStatus.AVAILABLE],
                unavailable_tick_count=counts[GeometryObservationStatus.UNAVAILABLE],
                invalid_tick_count=counts[GeometryObservationStatus.INVALID],
                interval_estimates=tuple(estimates),
                transition_matrices=transitions,
                retention=retention,
                history_qualified_reentry=reentry,
                decision=ViewQualitativeDecision(
                    view,
                    operational,
                    numerical,
                    support,
                    retention_pass,
                    reentry_pass,
                    expected,
                ),
            )
        )
    primary, fine = reports
    terminal = adjudicate_two_views(primary.decision, fine.decision)
    evaluation = MatrixObservationOrderEvaluation(
        evaluation_id=f"evaluation.{analysis_identity}",
        analysis_identity=analysis_identity,
        primary_report=ObjectIdentity.from_record(primary.report_id, primary),
        fine_report=ObjectIdentity.from_record(fine.report_id, fine),
        physical_history_count=config.expected_physical_histories,
        nested_view_count=config.expected_traces,
        qualification_fixture_id=config.qualification_fixture_id,
        terminal=terminal,
        reason_codes=(terminal.value,),
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    return primary, fine, evaluation


__all__ = [
    'MatrixObservationOrderEvaluationConfig',
    'MatrixObservationOrderEvaluation',
    'MatrixObservationOrderIntervalEstimate',
    'MatrixObservationOrderNonparametricMaximumLikelihoodPoint',
    'MatrixObservationOrderProjectionConfig',
    'MatrixObservationOrderTransitionCell',
    'MatrixObservationOrderTransitionMatrix',
    'MatrixObservationOrderViewReport',
    'evaluate_observation_order',
    'project_observation_order_history_view',
]
