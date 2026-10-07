"""Strict path-free scientific contracts for the independent substrate grounding Grid2Op target."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
import re
from typing import ClassVar

from empirical_lawhood.adapters.methods.structured_target import IndependentSubstrateTargetPhase
from empirical_lawhood.kernel.evidence import OutcomeAccess
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


class Grid2OpActionKind(StrEnum):
    HOLD = "HOLD"
    SET_LINE_STATUS = "SET_LINE_STATUS"
    SET_BUS = "SET_BUS"
    CHANGE_BUS = "CHANGE_BUS"
    REDISPATCH = "REDISPATCH"
    STORAGE = "STORAGE"
    CURTAILMENT = "CURTAILMENT"


@dataclass(frozen=True, slots=True)
class Grid2OpChronicBinding(CanonicalRecord):
    """One exact exogenous episode/chronic identity."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-chronic-binding'

    chronic_id: str
    native_chronic_id: str
    content_sha256: str
    initial_timestamp_utc: str
    timestep_seconds: int
    maximum_steps: int
    exogenous_event_sha256: str
    forecasts_available: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.chronic_id, field_name="chronic_id")
        validate_nonempty(self.native_chronic_id, field_name="native_chronic_id")
        validate_sha256(self.content_sha256, field_name="content_sha256")
        validate_sha256(
            self.exogenous_event_sha256,
            field_name="exogenous_event_sha256",
        )
        if (
            re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", self.initial_timestamp_utc)
            is None
        ):
            raise ValueError("Grid2Op chronic timestamp must be canonical UTC seconds")
        if self.timestep_seconds <= 0 or self.maximum_steps <= 0:
            raise ValueError("Grid2Op chronic timestep/length must be positive")


@dataclass(frozen=True, slots=True)
class Grid2OpSourceBinding(CanonicalRecord):
    """Exact package/backend/grid/rules/chronics/observation denominator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-source-binding'

    binding_id: str
    grid2op_version: str
    package_source_sha256: str
    environment_name: str
    backend_class_id: str
    backend_version: str
    backend_source_sha256: str
    grid_sha256: str
    rules_class_id: str
    parameters_sha256: str
    action_class_id: str
    observation_class_id: str
    observation_operator_sha256: str
    chronic_handler_class_id: str
    chronic_bindings: tuple[Grid2OpChronicBinding, ...]
    reward_ignored: bool
    source_network_required: bool
    fresh_target_identity_disjoint: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        for name in (
            "grid2op_version",
            "environment_name",
            "backend_class_id",
            "backend_version",
            "rules_class_id",
            "action_class_id",
            "observation_class_id",
            "chronic_handler_class_id",
        ):
            validate_nonempty(getattr(self, name), field_name=name)
        for name in (
            "package_source_sha256",
            "backend_source_sha256",
            "grid_sha256",
            "parameters_sha256",
            "observation_operator_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.chronic_bindings,
            attribute="chronic_id",
            field_name="chronic_bindings",
        )
        if not self.chronic_bindings:
            raise ValueError("Grid2Op source binding requires at least one chronic")
        native_ids = tuple(value.native_chronic_id for value in self.chronic_bindings)
        if len(set(native_ids)) != len(native_ids):
            raise ValueError("Grid2Op native chronic identities must be unique")
        if not self.reward_ignored:
            raise ValueError("independent substrate grounding Grid2Op is non-RL and must ignore reward")
        if self.source_network_required:
            raise ValueError("issued Grid2Op execution must use held offline source")
        # An exposed held chronic may be inspected in a development diagnostic.
        # Prospective design and sealed execution reject this flag separately.
        if type(self.fresh_target_identity_disjoint) is not bool:
            raise TypeError("Grid2Op source disjointness declaration must be boolean")


@dataclass(frozen=True, slots=True)
class Grid2OpNativeAction(CanonicalRecord):
    """Finite target-native action; no callable or free-form payload is allowed."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-native-action'

    action_id: str
    kind: Grid2OpActionKind
    target_native_ids: tuple[str, ...]
    integer_values: tuple[int, ...]
    decimal_values: tuple[Decimal, ...]
    canonical_description_bits: int

    def __post_init__(self) -> None:
        validate_stable_id(self.action_id, field_name="action_id")
        require_sorted_unique_strings(
            self.target_native_ids,
            field_name="target_native_ids",
        )
        for value in self.decimal_values:
            validate_decimal(value, field_name="decimal_values")
        if self.canonical_description_bits <= 0:
            raise ValueError("Grid2Op action description length must be positive")
        if self.kind is Grid2OpActionKind.HOLD:
            if self.target_native_ids or self.integer_values or self.decimal_values:
                raise ValueError("Grid2Op hold action cannot carry target values")
        elif not self.target_native_ids:
            raise ValueError("non-hold Grid2Op action requires native targets")
        if self.kind in {
            Grid2OpActionKind.SET_LINE_STATUS,
            Grid2OpActionKind.SET_BUS,
            Grid2OpActionKind.CHANGE_BUS,
        }:
            if len(self.integer_values) != len(self.target_native_ids) or self.decimal_values:
                raise ValueError("Grid2Op discrete action value roster differs")
        elif self.kind in {
            Grid2OpActionKind.REDISPATCH,
            Grid2OpActionKind.STORAGE,
            Grid2OpActionKind.CURTAILMENT,
        }:
            if len(self.decimal_values) != len(self.target_native_ids) or self.integer_values:
                raise ValueError("Grid2Op continuous action value roster differs")

    @property
    def action_code(self) -> str:
        return self.kind.value if self.kind is Grid2OpActionKind.HOLD else self.action_id


@dataclass(frozen=True, slots=True)
class Grid2OpActionBranchRequest(CanonicalRecord):
    """One action branch nested within a chronic independent unit."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-action-branch-request'

    branch_id: str
    action: Grid2OpNativeAction
    receiver_horizon_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.branch_id, field_name="branch_id")
        if (
            not self.receiver_horizon_steps
            or tuple(sorted(set(self.receiver_horizon_steps))) != self.receiver_horizon_steps
            or self.receiver_horizon_steps[0] <= 0
        ):
            raise ValueError("Grid2Op branch horizons must be sorted positive steps")


@dataclass(frozen=True, slots=True)
class Grid2OpEpisodeRequest(CanonicalRecord):
    """One issued chronic unit with reset/replay action branches as nested views."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-episode-request'

    request_id: str
    unit_id: str
    phase: IndependentSubstrateTargetPhase
    source_binding: ObjectIdentity
    chronic: ObjectIdentity
    environment_seed: int
    initial_step: int
    branches: tuple[Grid2OpActionBranchRequest, ...]
    receiver_gauge_ids: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")
        validate_stable_id(self.unit_id, field_name="unit_id")
        if self.source_binding.object_schema != Grid2OpSourceBinding.SCHEMA:
            raise ValueError("Grid2Op episode source identity differs")
        if self.chronic.object_schema != Grid2OpChronicBinding.SCHEMA:
            raise ValueError("Grid2Op episode chronic identity differs")
        if self.unit_id != self.chronic.object_id:
            raise ValueError("Grid2Op scientific unit must equal its chronic identity")
        if self.environment_seed < 0 or self.initial_step < 0:
            raise ValueError("Grid2Op seed/initial step must be nonnegative")
        require_sorted_unique_ids(self.branches, attribute="branch_id", field_name="branches")
        if not self.branches or not any(
            value.action.kind is Grid2OpActionKind.HOLD for value in self.branches
        ):
            raise ValueError("Grid2Op episode requires action branches and mandatory hold")
        require_sorted_unique_strings(
            self.receiver_gauge_ids,
            field_name="receiver_gauge_ids",
            allow_empty=False,
        )
        if set(self.receiver_gauge_ids) != {"thermal", "topology"}:
            raise ValueError("Grid2Op receiver gauge roster differs")
        expected_access = {
            IndependentSubstrateTargetPhase.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            IndependentSubstrateTargetPhase.EVALUATION: OutcomeAccess.EVALUATION_SEALED,
            IndependentSubstrateTargetPhase.PROSPECTIVE_VALIDATION: OutcomeAccess.EVALUATION_SEALED,
        }[self.phase]
        if self.outcome_access is not expected_access:
            raise ValueError("Grid2Op episode access differs from phase")


@dataclass(frozen=True, slots=True)
class Grid2OpNativeStep(CanonicalRecord):
    """One path-free result of an environment step at a declared horizon."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-native-step'

    step_id: str
    branch_id: str
    horizon_step: int
    environment_tick: int
    timestamp_utc: str
    requested_action_code: str
    accepted_action_code: str
    applied_action_code: str
    realized_action_code: str | None
    action_illegal: bool
    action_ambiguous: bool
    realization_observed: bool
    topology_state_id: str
    topology_vector_sha256: str
    connected_component_count: int
    disconnected_line_count: int
    maximum_rho: Decimal | None
    done: bool
    has_error: bool
    observation_valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("step_id", "branch_id", "topology_state_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.horizon_step <= 0 or self.environment_tick < 0:
            raise ValueError("Grid2Op horizon/tick is invalid")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", self.timestamp_utc) is None:
            raise ValueError("Grid2Op step timestamp must be canonical UTC seconds")
        for name in (
            "requested_action_code",
            "accepted_action_code",
            "applied_action_code",
        ):
            validate_nonempty(getattr(self, name), field_name=name)
        if self.realized_action_code is not None:
            validate_nonempty(self.realized_action_code, field_name="realized_action_code")
        validate_sha256(self.topology_vector_sha256, field_name="topology_vector_sha256")
        if self.connected_component_count <= 0 or self.disconnected_line_count < 0:
            raise ValueError("Grid2Op topology counts are invalid")
        if self.maximum_rho is not None:
            validate_decimal(self.maximum_rho, field_name="maximum_rho", minimum=Decimal(0))
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        replaced = self.action_illegal or self.action_ambiguous
        if replaced and (self.accepted_action_code != "HOLD" or self.applied_action_code != "HOLD"):
            raise ValueError("illegal/ambiguous Grid2Op action must be replaced by hold")
        if replaced and self.realization_observed and self.realized_action_code != "HOLD":
            raise ValueError("replacement Grid2Op action must realize the hold")
        if self.realization_observed != (self.realized_action_code is not None):
            raise ValueError("Grid2Op realization flag/code differ")
        if self.observation_valid:
            if self.maximum_rho is None or self.reason_codes:
                raise ValueError("valid Grid2Op observation lacks a receiver or has reasons")
        elif not self.reason_codes:
            raise ValueError("invalid Grid2Op observation requires a typed reason")
        if self.has_error and not self.done:
            raise ValueError("Grid2Op backend error must terminate the branch")


@dataclass(frozen=True, slots=True)
class Grid2OpBranchTrace(CanonicalRecord):
    """Exact reset-to-horizon trace for one nested action branch."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-branch-trace'

    trace_id: str
    request: ObjectIdentity
    branch: ObjectIdentity
    chronic: ObjectIdentity
    reset_state_sha256: str
    reset_exact: bool
    steps: tuple[Grid2OpNativeStep, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.trace_id, field_name="trace_id")
        if self.request.object_schema != Grid2OpEpisodeRequest.SCHEMA:
            raise ValueError("Grid2Op trace request identity differs")
        if self.branch.object_schema != Grid2OpActionBranchRequest.SCHEMA:
            raise ValueError("Grid2Op trace branch identity differs")
        if self.chronic.object_schema != Grid2OpChronicBinding.SCHEMA:
            raise ValueError("Grid2Op trace chronic identity differs")
        validate_sha256(self.reset_state_sha256, field_name="reset_state_sha256")
        step_ids = tuple(value.step_id for value in self.steps)
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("Grid2Op trace step identities must be unique")
        if not self.reset_exact or not self.steps:
            raise ValueError("Grid2Op trace requires exact reset and observed steps")
        horizons = tuple(value.horizon_step for value in self.steps)
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("Grid2Op trace horizons must be sorted and unique")
        if any(value.branch_id != self.branch.object_id for value in self.steps):
            raise ValueError("Grid2Op trace step branch identity differs")


@dataclass(frozen=True, slots=True)
class Grid2OpEpisodeResult(CanonicalRecord):
    """One complete chronic unit; branch traces remain nested views."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-episode-result'

    result_id: str
    request: ObjectIdentity
    source_binding: ObjectIdentity
    chronic: ObjectIdentity
    traces: tuple[Grid2OpBranchTrace, ...]
    scientific_unit_count: int
    reward_used_for_science: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.request.object_schema != Grid2OpEpisodeRequest.SCHEMA:
            raise ValueError("Grid2Op result request identity differs")
        if self.source_binding.object_schema != Grid2OpSourceBinding.SCHEMA:
            raise ValueError("Grid2Op result source identity differs")
        if self.chronic.object_schema != Grid2OpChronicBinding.SCHEMA:
            raise ValueError("Grid2Op result chronic identity differs")
        require_sorted_unique_ids(self.traces, attribute="trace_id", field_name="traces")
        if not self.traces or self.scientific_unit_count != 1:
            raise ValueError("one Grid2Op chronic result must count as one unit")
        if self.reward_used_for_science:
            raise ValueError("Grid2Op reward cannot enter independent substrate grounding science")


__all__ = [
    'Grid2OpActionBranchRequest',
    'Grid2OpActionKind',
    'Grid2OpBranchTrace',
    'Grid2OpChronicBinding',
    'Grid2OpEpisodeRequest',
    'Grid2OpEpisodeResult',
    'Grid2OpNativeAction',
    'Grid2OpNativeStep',
    'Grid2OpSourceBinding',
]
