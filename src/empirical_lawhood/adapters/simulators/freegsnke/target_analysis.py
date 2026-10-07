"""Target-native complete-preparation reduction for independent substrate grounding FreeGSNKE.

The reducer consumes exact issued requests and their outcome-capable response
envelopes.  It never drops an issued branch or preparation: missing, timed-out,
invalid, clipped, wrong-clock and mismatched actions remain typed outcomes.
Only preparations with the complete frozen nested branch/grid fibre are later
eligible for complete-unit inference; branches and receiver points never become
replicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import QuantityBound
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .contracts import FREEGSNKE_ACTIVE_CIRCUIT_IDS, FREEGSNKE_PASSIVE_CURRENT_SUMMARY_IDS, FREEGSNKE_PORT_IDS, FREEGSNKE_RECEIVER_UNITS, FreeGsnkeBranchKind, FreeGsnkePhase, FreeGsnkeProcessRequest, FreeGsnkeSourceBinding, FreeGsnkeTargetEpisodeStatus, FreeGsnkeTargetProcessResponse
from .design import FreeGsnkeActionDesign, FreeGsnkePhaseRequestRoster
from .runtime import decode_freegsnke_target_response


class FreeGsnkeReductionGridContract(Protocol):
    """Structural grid contract shared by ordinary and conditional-prospective validation reducers."""

    @property
    def required_horizons_s(self) -> tuple[Decimal, ...]: ...

    @property
    def required_receiver_ids(self) -> tuple[str, ...]: ...

    @property
    def required_topology_id(self) -> str: ...

    @property
    def required_active_current_ids(self) -> tuple[str, ...]: ...

    @property
    def passive_current_summary_ids(self) -> tuple[str, ...]: ...

    @property
    def retain_plasma_current(self) -> bool: ...


class FreeGsnkeBranchOutcomeStatus(StrEnum):
    OBSERVED_COMPLETE = "OBSERVED_COMPLETE"
    OBSERVED_INVALID = "OBSERVED_INVALID"
    WORKER_FAILED = "WORKER_FAILED"
    WORKER_TIMED_OUT = "WORKER_TIMED_OUT"
    MISSING_RESPONSE = "MISSING_RESPONSE"


@dataclass(frozen=True, slots=True)
class FreeGsnkePreparationStratumBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-preparation-stratum-binding'

    binding_id: str
    preparation: ObjectIdentity
    stratum_id: str
    denominator_stratum_id: str
    history_stratum_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.stratum_id, field_name="stratum_id")
        validate_stable_id(
            self.denominator_stratum_id,
            field_name="denominator_stratum_id",
        )
        validate_stable_id(self.history_stratum_id, field_name="history_stratum_id")
        if self.preparation.object_schema != (
            'empirical-lawhood/simulators/freegsnke/free-gsnke-preparation'
        ):
            raise ValueError("FreeGSNKE stratum preparation identity differs")


@dataclass(frozen=True, slots=True)
class FreeGsnkeTargetReductionConfig(CanonicalRecord):
    """Outcome-blind exact grid and complete-unit reduction freeze."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-target-reduction-config'

    config_id: str
    phase: FreeGsnkePhase
    phase_roster: ObjectIdentity
    action_design: ObjectIdentity
    source_binding: ObjectIdentity
    numerical_view: ObjectIdentity
    action_charts: tuple[ObjectIdentity, ...]
    hold_branch_id: str
    required_horizons_s: tuple[Decimal, ...]
    required_receiver_ids: tuple[str, ...]
    required_topology_id: str
    required_active_current_ids: tuple[str, ...]
    source_current_bounds: tuple[QuantityBound, ...]
    unbounded_active_current_ids: tuple[str, ...]
    passive_current_summary_ids: tuple[str, ...]
    retain_plasma_current: bool
    preparation_strata: tuple[FreeGsnkePreparationStratumBinding, ...]
    minimum_complete_units: int
    frozen_before_development: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.hold_branch_id, field_name="hold_branch_id")
        validate_stable_id(
            self.required_topology_id,
            field_name="required_topology_id",
        )
        expected_schemas = (
            (
                self.phase_roster,
                FreeGsnkePhaseRequestRoster.SCHEMA,
                "phase roster",
            ),
            (self.action_design, FreeGsnkeActionDesign.SCHEMA, "action design"),
            (self.source_binding, FreeGsnkeSourceBinding.SCHEMA, "source binding"),
            (
                self.numerical_view,
                'empirical-lawhood/simulators/freegsnke/free-gsnke-numerical-view',
                "numerical view",
            ),
        )
        for identity, schema, label in expected_schemas:
            if identity.object_schema != schema:
                raise ValueError(f"FreeGSNKE reduction {label} identity differs")
        require_sorted_unique_ids(
            self.action_charts,
            attribute="object_id",
            field_name="action_charts",
        )
        if not self.action_charts or any(
            value.object_schema != 'empirical-lawhood/simulators/freegsnke/free-gsnke-action-chart'
            for value in self.action_charts
        ):
            raise ValueError("FreeGSNKE reduction action-chart roster differs")
        if self.phase not in {FreeGsnkePhase.DEVELOPMENT, FreeGsnkePhase.EVALUATION}:
            raise ValueError("FreeGSNKE target reduction phase differs")
        for value in self.required_horizons_s:
            validate_decimal(value, field_name="required_horizons_s", minimum=Decimal(0))
        if (
            len(self.required_horizons_s) < 2
            or tuple(sorted(set(self.required_horizons_s))) != self.required_horizons_s
        ):
            raise ValueError("FreeGSNKE reduction horizons must be sorted and plural")
        require_sorted_unique_strings(
            self.required_receiver_ids,
            field_name="required_receiver_ids",
            allow_empty=False,
        )
        if self.required_receiver_ids != tuple(sorted(FREEGSNKE_RECEIVER_UNITS)):
            raise ValueError("FreeGSNKE target must retain the exact receiver roster")
        require_sorted_unique_strings(
            self.required_active_current_ids,
            field_name="required_active_current_ids",
            allow_empty=False,
        )
        if self.required_active_current_ids != FREEGSNKE_ACTIVE_CIRCUIT_IDS:
            raise ValueError("FreeGSNKE target must retain every active current")
        require_sorted_unique_ids(
            self.source_current_bounds,
            attribute="bound_id",
            field_name="source_current_bounds",
        )
        bounded_ids = set()
        for bound in self.source_current_bounds:
            if (
                not bound.quantity_id.endswith("-current")
                or bound.native_unit != "A"
                or bound.lower is None
                or bound.upper is None
            ):
                raise ValueError("FreeGSNKE source-current bound differs")
            bounded_ids.add(bound.quantity_id.removesuffix("-current"))
        require_sorted_unique_strings(
            self.unbounded_active_current_ids,
            field_name="unbounded_active_current_ids",
        )
        if (
            bounded_ids & set(self.unbounded_active_current_ids)
            or bounded_ids | set(self.unbounded_active_current_ids)
            != set(FREEGSNKE_ACTIVE_CIRCUIT_IDS)
            or not set(FREEGSNKE_PORT_IDS).issubset(bounded_ids)
        ):
            raise ValueError("FreeGSNKE bounded/unbounded active-current roles differ")
        require_sorted_unique_strings(
            self.passive_current_summary_ids,
            field_name="passive_current_summary_ids",
            allow_empty=False,
        )
        if self.passive_current_summary_ids != FREEGSNKE_PASSIVE_CURRENT_SUMMARY_IDS:
            raise ValueError("FreeGSNKE passive-current summary roster differs")
        if not self.retain_plasma_current:
            raise ValueError("FreeGSNKE target must retain realized plasma current")
        require_sorted_unique_ids(
            self.preparation_strata,
            attribute="binding_id",
            field_name="preparation_strata",
        )
        if len({value.preparation.object_id for value in self.preparation_strata}) != len(
            self.preparation_strata
        ):
            raise ValueError("FreeGSNKE preparation strata repeat a complete unit")
        if self.minimum_complete_units < 2:
            raise ValueError("FreeGSNKE target minimum complete-unit count differs")
        if not self.frozen_before_development or self.protected_outcome_access_count:
            raise ValueError("FreeGSNKE reduction config crossed its outcome-blind freeze")


@dataclass(frozen=True, slots=True)
class FreeGsnkeActionFibreAudit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-action-fibre-audit'

    audit_id: str
    branch_id: str
    status: FreeGsnkeBranchOutcomeStatus
    accepted_observed: bool
    acceptance_clock_observed: bool
    applied_observed: bool
    realized_observed: bool
    receiver_observed: bool
    causal_state_observed: bool
    requested_accepted_exact: bool
    accepted_applied_exact: bool
    clipping_observed: bool
    temporal_order_passed: bool
    receiver_realization_clocks_aligned: bool
    causal_cutoff_passed: bool
    action_ontology_error_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        validate_stable_id(self.branch_id, field_name="branch_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected_errors = sum(
            (
                self.accepted_observed and not self.requested_accepted_exact,
                self.accepted_observed != self.acceptance_clock_observed,
                self.applied_observed
                and self.accepted_observed
                and not self.accepted_applied_exact,
                self.clipping_observed,
                (self.applied_observed or self.realized_observed)
                and not self.acceptance_clock_observed,
                self.acceptance_clock_observed and not self.temporal_order_passed,
                self.receiver_observed
                and self.realized_observed
                and not self.receiver_realization_clocks_aligned,
                self.causal_state_observed and not self.causal_cutoff_passed,
            )
        )
        if self.action_ontology_error_count != expected_errors:
            raise ValueError("FreeGSNKE action-ontology error count is not fact-derived")

    @property
    def qualified(self) -> bool:
        return all(
            (
                self.accepted_observed,
                self.acceptance_clock_observed,
                self.applied_observed,
                self.realized_observed,
                self.receiver_observed,
                self.causal_state_observed,
                self.requested_accepted_exact,
                self.accepted_applied_exact,
                not self.clipping_observed,
                self.temporal_order_passed,
                self.receiver_realization_clocks_aligned,
                self.causal_cutoff_passed,
            )
        )


@dataclass(frozen=True, slots=True)
class FreeGsnkeNativePoint(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-native-point'

    point_id: str
    branch_id: str
    coordinate_id: str
    horizon_s: Decimal
    native_unit: str
    value: Decimal
    topology_id: str
    receiver_valid: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.point_id, field_name="point_id")
        validate_stable_id(self.branch_id, field_name="branch_id")
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        validate_stable_id(self.topology_id, field_name="topology_id")
        validate_decimal(self.horizon_s, field_name="horizon_s", minimum=Decimal(0))
        validate_decimal(self.value, field_name="value")
        if not self.native_unit:
            raise ValueError("FreeGSNKE native point requires its unit")


@dataclass(frozen=True, slots=True)
class FreeGsnkeBranchReduction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-branch-reduction'

    reduction_id: str
    branch_id: str
    branch_kind: FreeGsnkeBranchKind
    request: ObjectIdentity
    response: ObjectIdentity | None
    status: FreeGsnkeBranchOutcomeStatus
    action_audit: FreeGsnkeActionFibreAudit
    native_points: tuple[FreeGsnkeNativePoint, ...]
    analysis_grid_complete: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.reduction_id, field_name="reduction_id")
        validate_stable_id(self.branch_id, field_name="branch_id")
        if self.request.object_schema != FreeGsnkeProcessRequest.SCHEMA:
            raise ValueError("FreeGSNKE branch reduction request identity differs")
        if self.response is not None and self.response.object_schema != (
            FreeGsnkeTargetProcessResponse.SCHEMA
        ):
            raise ValueError("FreeGSNKE branch reduction response identity differs")
        if (self.response is None) != (
            self.status is FreeGsnkeBranchOutcomeStatus.MISSING_RESPONSE
        ):
            raise ValueError("FreeGSNKE missing-response status is not fact-derived")
        if (
            self.action_audit.branch_id != self.branch_id
            or self.action_audit.status is not self.status
        ):
            raise ValueError("FreeGSNKE branch/action audit differs")
        require_sorted_unique_ids(
            self.native_points,
            attribute="point_id",
            field_name="native_points",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class FreeGsnkeNativeContrast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-native-contrast'

    contrast_id: str
    branch_id: str
    baseline_branch_id: str
    coordinate_id: str
    horizon_s: Decimal
    native_unit: str
    branch_value: Decimal
    baseline_value: Decimal
    delta: Decimal

    def __post_init__(self) -> None:
        for name in (
            "contrast_id",
            "branch_id",
            "baseline_branch_id",
            "coordinate_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_decimal(self.horizon_s, field_name="horizon_s", minimum=Decimal(0))
        for name in ("branch_value", "baseline_value", "delta"):
            validate_decimal(getattr(self, name), field_name=name)
        if self.delta != self.branch_value - self.baseline_value:
            raise ValueError("FreeGSNKE native contrast delta is not value-derived")


@dataclass(frozen=True, slots=True)
class FreeGsnkePreparationReduction(CanonicalRecord):
    """One independent preparation with its entire nested issued branch fibre."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-preparation-reduction'

    reduction_id: str
    preparation: ObjectIdentity
    phase: FreeGsnkePhase
    stratum_id: str
    denominator_stratum_id: str
    history_stratum_id: str
    branches: tuple[FreeGsnkeBranchReduction, ...]
    contrasts: tuple[FreeGsnkeNativeContrast, ...]
    issued_branch_count: int
    observed_response_count: int
    complete_grid_branch_count: int
    action_qualified_branch_count: int
    action_ontology_error_count: int
    reason_codes: tuple[str, ...]

    @property
    def unit_id(self) -> str:
        return self.preparation.object_id

    def __post_init__(self) -> None:
        validate_stable_id(self.reduction_id, field_name="reduction_id")
        validate_stable_id(self.stratum_id, field_name="stratum_id")
        validate_stable_id(
            self.denominator_stratum_id,
            field_name="denominator_stratum_id",
        )
        validate_stable_id(self.history_stratum_id, field_name="history_stratum_id")
        if self.preparation.object_schema != (
            'empirical-lawhood/simulators/freegsnke/free-gsnke-preparation'
        ):
            raise ValueError("FreeGSNKE unit preparation identity differs")
        require_sorted_unique_ids(
            self.branches,
            attribute="branch_id",
            field_name="branches",
        )
        require_sorted_unique_ids(
            self.contrasts,
            attribute="contrast_id",
            field_name="contrasts",
        )
        expected = (
            len(self.branches),
            sum(value.response is not None for value in self.branches),
            sum(value.analysis_grid_complete for value in self.branches),
            sum(value.action_audit.qualified for value in self.branches),
            sum(value.action_audit.action_ontology_error_count for value in self.branches),
        )
        observed = (
            self.issued_branch_count,
            self.observed_response_count,
            self.complete_grid_branch_count,
            self.action_qualified_branch_count,
            self.action_ontology_error_count,
        )
        if observed != expected:
            raise ValueError("FreeGSNKE preparation counts are not branch-derived")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")

    @property
    def complete_for_inference(self) -> bool:
        return bool(self.branches) and all(
            value.analysis_grid_complete and value.action_audit.qualified for value in self.branches
        )


@dataclass(frozen=True, slots=True)
class FreeGsnkePhaseReduction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-phase-reduction'

    reduction_id: str
    config: ObjectIdentity
    phase_roster: ObjectIdentity
    phase: FreeGsnkePhase
    units: tuple[FreeGsnkePreparationReduction, ...]
    issued_unit_count: int
    inference_complete_unit_count: int
    postissue_adverse_unit_count: int
    missing_response_count: int
    action_ontology_error_count: int
    frozen_minimum_complete_units: int
    panel_envelope_limited: bool
    complete_unit_ids_sha256: str
    issued_unit_ids_sha256: str
    nested_observations_count_as_units: bool
    outcome_access: OutcomeAccess
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.reduction_id, field_name="reduction_id")
        if self.config.object_schema != FreeGsnkeTargetReductionConfig.SCHEMA:
            raise ValueError("FreeGSNKE phase reduction config identity differs")
        if self.phase_roster.object_schema != FreeGsnkePhaseRequestRoster.SCHEMA:
            raise ValueError("FreeGSNKE phase reduction roster identity differs")
        require_sorted_unique_ids(
            self.units,
            attribute="unit_id",
            field_name="units",
        )
        if any(value.phase is not self.phase for value in self.units):
            raise ValueError("FreeGSNKE phase reduction crosses phase partitions")
        expected_complete = sum(value.complete_for_inference for value in self.units)
        expected_missing = sum(
            value.issued_branch_count - value.observed_response_count for value in self.units
        )
        expected_errors = sum(value.action_ontology_error_count for value in self.units)
        if (
            self.issued_unit_count != len(self.units)
            or self.inference_complete_unit_count != expected_complete
            or self.postissue_adverse_unit_count != len(self.units) - expected_complete
            or self.missing_response_count != expected_missing
            or self.action_ontology_error_count != expected_errors
        ):
            raise ValueError("FreeGSNKE phase counts are not complete-unit-derived")
        if self.panel_envelope_limited != (
            self.inference_complete_unit_count < self.frozen_minimum_complete_units
        ):
            raise ValueError("FreeGSNKE panel limitation is not count-derived")
        validate_sha256(
            self.complete_unit_ids_sha256,
            field_name="complete_unit_ids_sha256",
        )
        validate_sha256(self.issued_unit_ids_sha256, field_name="issued_unit_ids_sha256")
        if self.nested_observations_count_as_units:
            raise ValueError("FreeGSNKE nested branches/points cannot inflate units")
        expected_access = {
            FreeGsnkePhase.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            FreeGsnkePhase.EVALUATION: OutcomeAccess.EVALUATION_SEALED,
        }[self.phase]
        if self.outcome_access is not expected_access:
            raise ValueError("FreeGSNKE phase reduction outcome access differs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


def _digest_ids(values: tuple[str, ...]) -> str:
    return sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()


def _branch_status(
    response: FreeGsnkeTargetProcessResponse | None,
) -> FreeGsnkeBranchOutcomeStatus:
    if response is None:
        return FreeGsnkeBranchOutcomeStatus.MISSING_RESPONSE
    return FreeGsnkeBranchOutcomeStatus(response.episode.status.value)


def _action_audit(
    *,
    request: FreeGsnkeProcessRequest,
    response: FreeGsnkeTargetProcessResponse | None,
    status: FreeGsnkeBranchOutcomeStatus,
) -> FreeGsnkeActionFibreAudit:
    if response is None:
        return FreeGsnkeActionFibreAudit(
            audit_id=f"audit.{request.request_id}",
            branch_id=request.branch_id,
            status=status,
            accepted_observed=False,
            acceptance_clock_observed=False,
            applied_observed=False,
            realized_observed=False,
            receiver_observed=False,
            causal_state_observed=False,
            requested_accepted_exact=False,
            accepted_applied_exact=False,
            clipping_observed=False,
            temporal_order_passed=False,
            receiver_realization_clocks_aligned=False,
            causal_cutoff_passed=False,
            action_ontology_error_count=0,
            reason_codes=("POSTISSUE_RESPONSE_MISSING",),
        )
    episode = response.episode
    action = episode.action_observation
    accepted = bool(action.accepted_port_increments)
    acceptance_clock = action.acceptance_clock_s is not None
    applied = bool(action.applied_samples)
    realized = bool(action.realized_currents)
    receiver = bool(episode.receiver_snapshots)
    causal = episode.causal_state is not None
    requested_accepted = accepted and (
        action.accepted_port_increments == request.requested_port_increments
    )
    accepted_applied = (
        accepted
        and applied
        and all(
            value.coil_voltage_increments == action.accepted_port_increments
            for value in action.applied_samples
        )
    )
    temporal = True
    if acceptance_clock:
        assert action.acceptance_clock_s is not None
        temporal = action.acceptance_clock_s >= action.request_clock_s
        if applied:
            temporal = temporal and all(
                value.clock_s >= action.acceptance_clock_s for value in action.applied_samples
            )
    if receiver:
        temporal = temporal and all(
            value.receiver_clock_s > action.request_clock_s for value in episode.receiver_snapshots
        )
    if realized:
        temporal = temporal and all(
            value.receiver_clock_s > action.request_clock_s for value in action.realized_currents
        )
    receiver_clocks = tuple(value.receiver_clock_s for value in episode.receiver_snapshots)
    realized_clocks = tuple(value.receiver_clock_s for value in action.realized_currents)
    aligned = receiver and realized and receiver_clocks == realized_clocks
    causal_passed = causal and (
        episode.causal_state is not None
        and episode.causal_state.cutoff_clock_s <= action.request_clock_s
    )
    errors = sum(
        (
            accepted and not requested_accepted,
            accepted != acceptance_clock,
            applied and accepted and not accepted_applied,
            bool(action.clipping_flags),
            (applied or realized) and not acceptance_clock,
            acceptance_clock and not temporal,
            receiver and realized and not aligned,
            causal and not causal_passed,
        )
    )
    reasons = {
        *(() if accepted else ("ACCEPTED_ACTION_MISSING",)),
        *(() if acceptance_clock else ("ACCEPTANCE_CLOCK_MISSING",)),
        *(() if applied else ("APPLIED_ACTION_MISSING",)),
        *(() if realized else ("REALIZED_ACTION_MISSING",)),
        *(() if receiver else ("RECEIVER_OBSERVATION_MISSING",)),
        *(() if causal else ("CAUSAL_STATE_MISSING",)),
        *(() if not accepted or requested_accepted else ("REQUEST_ACCEPT_MISMATCH",)),
        *(() if not applied or not accepted or accepted_applied else ("ACCEPT_APPLY_MISMATCH",)),
        *(("ACTION_CLIPPED",) if action.clipping_flags else ()),
        *(() if temporal else ("ACTION_CLOCK_ORDER_ERROR",)),
        *(
            ()
            if not receiver or not realized or aligned
            else ("RECEIVER_REALIZATION_CLOCK_MISMATCH",)
        ),
        *(() if not causal or causal_passed else ("CAUSAL_CUTOFF_CROSSED",)),
        *(() if status is FreeGsnkeBranchOutcomeStatus.OBSERVED_COMPLETE else (status.value,)),
    }
    return FreeGsnkeActionFibreAudit(
        audit_id=f"audit.{request.request_id}",
        branch_id=request.branch_id,
        status=status,
        accepted_observed=accepted,
        acceptance_clock_observed=acceptance_clock,
        applied_observed=applied,
        realized_observed=realized,
        receiver_observed=receiver,
        causal_state_observed=causal,
        requested_accepted_exact=requested_accepted,
        accepted_applied_exact=accepted_applied,
        clipping_observed=bool(action.clipping_flags),
        temporal_order_passed=temporal,
        receiver_realization_clocks_aligned=aligned,
        causal_cutoff_passed=causal_passed,
        action_ontology_error_count=errors,
        reason_codes=tuple(sorted(reasons)),
    )


def _native_points(
    *,
    request: FreeGsnkeProcessRequest,
    response: FreeGsnkeTargetProcessResponse | None,
    config: FreeGsnkeReductionGridContract,
) -> tuple[FreeGsnkeNativePoint, ...]:
    if response is None:
        return ()
    receivers = {value.receiver_clock_s: value for value in response.episode.receiver_snapshots}
    currents = {
        value.receiver_clock_s: value
        for value in response.episode.action_observation.realized_currents
    }
    values = []
    for horizon_index, horizon in enumerate(config.required_horizons_s):
        receiver = receivers.get(horizon)
        if receiver is not None:
            by_id = {value.value_id: value for value in receiver.values}
            for receiver_id in config.required_receiver_ids:
                value = by_id.get(receiver_id)
                if value is not None:
                    values.append(
                        FreeGsnkeNativePoint(
                            point_id=(
                                f"{request.request_id}.point.receiver."
                                f"{receiver_id}.h{horizon_index:02d}"
                            ),
                            branch_id=request.branch_id,
                            coordinate_id=receiver_id,
                            horizon_s=horizon,
                            native_unit=value.unit,
                            value=value.value,
                            topology_id=receiver.topology_id,
                            receiver_valid=receiver.valid,
                        )
                    )
        current = currents.get(horizon)
        if current is not None:
            by_id = {value.value_id: value for value in current.active_currents}
            receiver_topology = (
                receiver.topology_id if receiver is not None else "receiver-topology-unobserved"
            )
            receiver_valid = receiver is not None and receiver.valid
            for circuit_id in config.required_active_current_ids:
                value = by_id.get(circuit_id)
                if value is not None:
                    values.append(
                        FreeGsnkeNativePoint(
                            point_id=(
                                f"{request.request_id}.point.current-"
                                f"{circuit_id}.h{horizon_index:02d}"
                            ),
                            branch_id=request.branch_id,
                            coordinate_id=f"current-{circuit_id}",
                            horizon_s=horizon,
                            native_unit=value.unit,
                            value=value.value,
                            topology_id=receiver_topology,
                            receiver_valid=receiver_valid,
                        )
                    )
            if current.plasma_current is not None:
                values.append(
                    FreeGsnkeNativePoint(
                        point_id=(f"{request.request_id}.point.current-ip.h{horizon_index:02d}"),
                        branch_id=request.branch_id,
                        coordinate_id="current-ip",
                        horizon_s=horizon,
                        native_unit=current.plasma_current.unit,
                        value=current.plasma_current.value,
                        topology_id=receiver_topology,
                        receiver_valid=receiver_valid,
                    )
                )
            if len(current.passive_currents) == 138:
                passive_values = tuple(value.value for value in current.passive_currents)
                summaries = {
                    "passive-current-l1": sum(
                        (abs(value) for value in passive_values),
                        start=Decimal(0),
                    ),
                    "passive-current-max-abs": max(abs(value) for value in passive_values),
                    "passive-current-signed-sum": sum(
                        passive_values,
                        start=Decimal(0),
                    ),
                }
                for summary_id in config.passive_current_summary_ids:
                    values.append(
                        FreeGsnkeNativePoint(
                            point_id=(
                                f"{request.request_id}.point.{summary_id}.h{horizon_index:02d}"
                            ),
                            branch_id=request.branch_id,
                            coordinate_id=summary_id,
                            horizon_s=horizon,
                            native_unit="A",
                            value=summaries[summary_id],
                            topology_id=receiver_topology,
                            receiver_valid=receiver_valid,
                        )
                    )
    return tuple(sorted(values, key=lambda value: value.point_id))


def _grid_complete(
    *,
    points: tuple[FreeGsnkeNativePoint, ...],
    response: FreeGsnkeTargetProcessResponse | None,
    config: FreeGsnkeReductionGridContract,
) -> bool:
    expected_coordinates = (
        len(config.required_receiver_ids)
        + len(config.required_active_current_ids)
        + len(config.passive_current_summary_ids)
        + int(config.retain_plasma_current)
    )
    return bool(
        response is not None
        and response.episode.status is FreeGsnkeTargetEpisodeStatus.OBSERVED_COMPLETE
        and len(points) == len(config.required_horizons_s) * expected_coordinates
        and all(
            value.receiver_valid and value.topology_id == config.required_topology_id
            for value in points
        )
    )


def _contrasts(
    *,
    branches: tuple[FreeGsnkeBranchReduction, ...],
    hold_branch_id: str,
) -> tuple[FreeGsnkeNativeContrast, ...]:
    by_branch = {value.branch_id: value for value in branches}
    hold = by_branch[hold_branch_id]
    baseline = {(value.coordinate_id, value.horizon_s): value for value in hold.native_points}
    values = []
    for branch in branches:
        if branch.branch_id == hold_branch_id:
            continue
        for point in branch.native_points:
            base = baseline.get((point.coordinate_id, point.horizon_s))
            if base is None or base.native_unit != point.native_unit:
                continue
            horizon_index = tuple(
                sorted({value.horizon_s for value in branch.native_points})
            ).index(point.horizon_s)
            values.append(
                FreeGsnkeNativeContrast(
                    contrast_id=(
                        f"contrast.{branch.request.object_id}.{point.coordinate_id}."
                        f"h{horizon_index:02d}"
                    ),
                    branch_id=branch.branch_id,
                    baseline_branch_id=hold_branch_id,
                    coordinate_id=point.coordinate_id,
                    horizon_s=point.horizon_s,
                    native_unit=point.native_unit,
                    branch_value=point.value,
                    baseline_value=base.value,
                    delta=point.value - base.value,
                )
            )
    return tuple(sorted(values, key=lambda value: value.contrast_id))


def reduce_freegsnke_subset_units(
    *,
    config: FreeGsnkeReductionGridContract,
    phase: FreeGsnkePhase,
    preparation_identities: tuple[ObjectIdentity, ...],
    preparation_strata: tuple[FreeGsnkePreparationStratumBinding, ...],
    expected_branch_ids: tuple[str, ...],
    hold_branch_id: str,
    requests: tuple[FreeGsnkeProcessRequest, ...],
    responses: tuple[FreeGsnkeTargetProcessResponse, ...],
) -> tuple[FreeGsnkePreparationReduction, ...]:
    """Reduce an exact issued branch subset without changing its unit denominator.

    Development/evaluation pass the complete frozen action chart.  Conditional
    prospective validation passes only the parent-admission evaluation selected action plus measured hold (or hold
    alone).  The action audit, native observation grid and adverse-outcome
    retention are consequently identical across all three stages.
    """

    require_sorted_unique_strings(
        expected_branch_ids,
        field_name="expected_branch_ids",
        allow_empty=False,
    )
    if hold_branch_id not in expected_branch_ids:
        raise ValueError("FreeGSNKE reduced branch subset requires measured hold")
    ordered_requests = tuple(sorted(requests, key=lambda value: value.request_id))
    if len({value.request_id for value in ordered_requests}) != len(ordered_requests):
        raise ValueError("FreeGSNKE reduced request identities repeat")
    request_by_id = {value.request_id: value for value in ordered_requests}
    response_by_request_id: dict[str, FreeGsnkeTargetProcessResponse] = {}
    for candidate in responses:
        bound_request = request_by_id.get(candidate.request.object_id)
        if bound_request is None:
            raise ValueError("FreeGSNKE response is outside the issued request roster")
        bound_response = decode_freegsnke_target_response(
            request=bound_request,
            payload=candidate.canonical_bytes(),
        )
        if bound_request.request_id in response_by_request_id:
            raise ValueError("FreeGSNKE issued request has multiple responses")
        response_by_request_id[bound_request.request_id] = bound_response

    stratum_by_preparation = {value.preparation.object_id: value for value in preparation_strata}
    preparation_ids = tuple(value.object_id for value in preparation_identities)
    if set(stratum_by_preparation) != set(preparation_ids):
        raise ValueError("FreeGSNKE reduction strata differ from the issued units")
    expected = set(expected_branch_ids)
    units = []
    for preparation_identity in preparation_identities:
        local_requests = tuple(
            value
            for value in ordered_requests
            if value.preparation.preparation_id == preparation_identity.object_id
        )
        if (
            {value.branch_id for value in local_requests} != expected
            or len(local_requests) != len(expected)
            or any(value.phase is not phase for value in local_requests)
        ):
            raise ValueError("FreeGSNKE reduction branch subset differs from its issue")
        branches = []
        for request in sorted(local_requests, key=lambda value: value.branch_id):
            branch_response = response_by_request_id.get(request.request_id)
            status = _branch_status(branch_response)
            audit = _action_audit(
                request=request,
                response=branch_response,
                status=status,
            )
            points = _native_points(
                request=request,
                response=branch_response,
                config=config,
            )
            grid_complete = _grid_complete(
                points=points,
                response=branch_response,
                config=config,
            )
            reasons = {
                *audit.reason_codes,
                *(() if grid_complete else ("FROZEN_ANALYSIS_GRID_INCOMPLETE",)),
            }
            branches.append(
                FreeGsnkeBranchReduction(
                    reduction_id=f"reduction.{request.request_id}",
                    branch_id=request.branch_id,
                    branch_kind=request.branch_kind,
                    request=ObjectIdentity.from_record(request.request_id, request),
                    response=(
                        ObjectIdentity.from_record(
                            branch_response.response_id,
                            branch_response,
                        )
                        if branch_response is not None
                        else None
                    ),
                    status=status,
                    action_audit=audit,
                    native_points=points,
                    analysis_grid_complete=grid_complete,
                    reason_codes=tuple(sorted(reasons)),
                )
            )
        ordered_branches = tuple(sorted(branches, key=lambda value: value.branch_id))
        contrasts = _contrasts(
            branches=ordered_branches,
            hold_branch_id=hold_branch_id,
        )
        unit_reasons = {reason for branch in ordered_branches for reason in branch.reason_codes}
        stratum = stratum_by_preparation[preparation_identity.object_id]
        units.append(
            FreeGsnkePreparationReduction(
                reduction_id=f"reduction.{preparation_identity.object_id}",
                preparation=preparation_identity,
                phase=phase,
                stratum_id=stratum.stratum_id,
                denominator_stratum_id=stratum.denominator_stratum_id,
                history_stratum_id=stratum.history_stratum_id,
                branches=ordered_branches,
                contrasts=contrasts,
                issued_branch_count=len(ordered_branches),
                observed_response_count=sum(
                    value.response is not None for value in ordered_branches
                ),
                complete_grid_branch_count=sum(
                    value.analysis_grid_complete for value in ordered_branches
                ),
                action_qualified_branch_count=sum(
                    value.action_audit.qualified for value in ordered_branches
                ),
                action_ontology_error_count=sum(
                    value.action_audit.action_ontology_error_count for value in ordered_branches
                ),
                reason_codes=tuple(sorted(unit_reasons)),
            )
        )
    return tuple(sorted(units, key=lambda value: value.preparation.object_id))


def reduce_freegsnke_phase_panel(
    *,
    config: FreeGsnkeTargetReductionConfig,
    roster: FreeGsnkePhaseRequestRoster,
    action_design: FreeGsnkeActionDesign,
    requests: tuple[FreeGsnkeProcessRequest, ...],
    responses: tuple[FreeGsnkeTargetProcessResponse, ...],
) -> FreeGsnkePhaseReduction:
    """Reduce one phase without changing its issued complete-unit denominator."""

    roster_identity = ObjectIdentity.from_record(roster.roster_id, roster)
    design_identity = ObjectIdentity.from_record(action_design.design_id, action_design)
    if (
        config.phase_roster != roster_identity
        or config.action_design != design_identity
        or roster.action_design != design_identity
        or config.phase is not roster.phase
    ):
        raise ValueError("FreeGSNKE reduction design/roster freeze differs")
    branch_by_id = {value.branch_id: value for value in action_design.branches}
    try:
        hold = branch_by_id[config.hold_branch_id]
    except KeyError as error:
        raise ValueError("FreeGSNKE reduction hold branch is outside the design") from error
    if hold.branch_kind is not FreeGsnkeBranchKind.COMPARATOR:
        raise ValueError("FreeGSNKE reduction baseline must be an exact hold branch")

    ordered_requests = tuple(sorted(requests, key=lambda value: value.request_id))
    request_identities = tuple(
        ObjectIdentity.from_record(value.request_id, value) for value in ordered_requests
    )
    if request_identities != roster.requests:
        raise ValueError("FreeGSNKE reduction request bytes differ from the issued roster")
    for request in ordered_requests:
        if (
            request.phase is not config.phase
            or ObjectIdentity.from_record(
                request.source_binding.binding_id,
                request.source_binding,
            )
            != config.source_binding
            or ObjectIdentity.from_record(
                request.numerical_view.view_id,
                request.numerical_view,
            )
            != config.numerical_view
            or ObjectIdentity.from_record(
                request.action_chart.chart_id,
                request.action_chart,
            )
            not in config.action_charts
        ):
            raise ValueError("FreeGSNKE reduction request changes a frozen operand")
    chart_horizon_rosters = {
        tuple(
            sorted(
                {
                    *request.action_chart.primary_horizons_s,
                    *request.action_chart.diagnostic_horizons_s,
                }
            )
        )
        for request in ordered_requests
    }
    if chart_horizon_rosters != {config.required_horizons_s}:
        raise ValueError("FreeGSNKE reduction must retain the exact chart horizon roster")

    ordered_units = reduce_freegsnke_subset_units(
        config=config,
        phase=config.phase,
        preparation_identities=roster.preparations,
        preparation_strata=config.preparation_strata,
        expected_branch_ids=tuple(sorted(branch_by_id)),
        hold_branch_id=config.hold_branch_id,
        requests=ordered_requests,
        responses=responses,
    )
    complete_ids = tuple(
        value.preparation.object_id for value in ordered_units if value.complete_for_inference
    )
    issued_ids = tuple(value.preparation.object_id for value in ordered_units)
    inference_count = len(complete_ids)
    reasons = set()
    if inference_count < config.minimum_complete_units:
        reasons.add("PANEL_ENVELOPE_LIMITED")
    if any(value.action_ontology_error_count for value in ordered_units):
        reasons.add("ACTION_ONTOLOGY_COUNTEREXAMPLE")
    if any(value.observed_response_count < value.issued_branch_count for value in ordered_units):
        reasons.add("POSTISSUE_RESPONSE_MISSING")
    return FreeGsnkePhaseReduction(
        reduction_id=f"reduction.independent-substrate-grounding-freegsnke.{config.phase.value.lower()}",
        config=ObjectIdentity.from_record(config.config_id, config),
        phase_roster=roster_identity,
        phase=config.phase,
        units=ordered_units,
        issued_unit_count=len(ordered_units),
        inference_complete_unit_count=inference_count,
        postissue_adverse_unit_count=len(ordered_units) - inference_count,
        missing_response_count=sum(
            value.issued_branch_count - value.observed_response_count for value in ordered_units
        ),
        action_ontology_error_count=sum(
            value.action_ontology_error_count for value in ordered_units
        ),
        frozen_minimum_complete_units=config.minimum_complete_units,
        panel_envelope_limited=inference_count < config.minimum_complete_units,
        complete_unit_ids_sha256=_digest_ids(complete_ids),
        issued_unit_ids_sha256=_digest_ids(issued_ids),
        nested_observations_count_as_units=False,
        outcome_access=(
            OutcomeAccess.DEVELOPMENT_VISIBLE
            if config.phase is FreeGsnkePhase.DEVELOPMENT
            else OutcomeAccess.EVALUATION_SEALED
        ),
        reason_codes=tuple(sorted(reasons)),
    )


__all__ = [
    'FreeGsnkeActionFibreAudit',
    'FreeGsnkeBranchOutcomeStatus',
    'FreeGsnkeBranchReduction',
    'FreeGsnkeNativeContrast',
    'FreeGsnkeNativePoint',
    'FreeGsnkePhaseReduction',
    'FreeGsnkePreparationReduction',
    'FreeGsnkePreparationStratumBinding',
    "FreeGsnkeReductionGridContract",
    'FreeGsnkeTargetReductionConfig',
    "reduce_freegsnke_phase_panel",
    "reduce_freegsnke_subset_units",
]
