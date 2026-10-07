"""Direct complete-history acquisition for the Six-matrix response prospective source law.

One acquisition owns one independently seeded preparation history and its
predeclared direct-future panel.  A frozen routing service may inspect only the
completed history before future streams are derived.  Scientific projection,
model fitting, and terminal reduction remain method-owned.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from io import BytesIO
from typing import Any, ClassVar

import numpy as np

from empirical_lawhood.adapters.simulators.six_matrix_response.reactive_source_scientific_inputs import MatrixReactiveSourceHistoryScientificInput, MatrixReactiveSourceRoutingScientificInput, original_routing_scientific_input_sha256, reactive_source_history_scientific_inputs
from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView
from empirical_lawhood.adapters.simulators.six_matrix_response.history_preparation import FOUR_FAMILY_PREPARATION_IDS, four_family_preparation_couplings
from empirical_lawhood.adapters.simulators.six_matrix_response.model import ComplexArray, SixMatrixState, hermiticity_residual, ideal_state
from empirical_lawhood.adapters.simulators.six_matrix_response.shooting import brownian_bridge_split
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import BAOABGradientCache, baoab_step_with_hermitian_noise, hermitian_noise
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.response_experiment_ports import NativeAcquisitionRequest

SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA = 'empirical-lawhood/simulators/six-matrix-response/reactive-source-history-future-panel-hdf5'
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_MEDIA_TYPE = "application/x-hdf5"
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANONICAL_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_KEYS = (
    '["/future/arm","/future/checkpoint_step","/future/future_index",'
    '"/future/innovations","/future/momenta","/future/positions",'
    '"/history/alpha_tilde","/history/innovations","/history/momenta",'
    '"/history/positions","/history/step_index","/routing/alpha_tilde",'
    '"/routing/momenta","/routing/positions"]'
)
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_UNITS = (
    "{"
    + ",".join(
        f'"{path}":"{unit}"'
        for path, unit in (
            ("/future/arm", "categorical-arm"),
            ("/future/checkpoint_step", "integration-step"),
            ("/future/future_index", "direct-future-index"),
            ("/future/innovations", "standard-normal-complex-hermitian"),
            ("/future/momenta", "dimensionless-canonical-momentum"),
            ("/future/positions", "dimensionless-matrix-coordinate"),
            ("/history/alpha_tilde", "dimensionless-scaled-coupling"),
            ("/history/innovations", "standard-normal-complex-hermitian"),
            ("/history/momenta", "dimensionless-canonical-momentum"),
            ("/history/positions", "dimensionless-matrix-coordinate"),
            ("/history/step_index", "integration-step"),
            ("/routing/alpha_tilde", "dimensionless-scaled-coupling"),
            ("/routing/momenta", "dimensionless-canonical-momentum"),
            ("/routing/positions", "dimensionless-matrix-coordinate"),
        )
    )
    + "}"
)
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_FRAMES = (
    "{"
    + ",".join(
        f'"{path}":"{frame}"'
        for path, frame in (
            ("/future/arm", "six-matrix-response.frame.routing-ledger"),
            ("/future/checkpoint_step", "six-matrix-response.frame.integration-index"),
            ("/future/future_index", "six-matrix-response.frame.routing-ledger"),
            ("/future/innovations", "six-matrix-response.frame.six-matrix-hermitian"),
            ("/future/momenta", "six-matrix-response.frame.six-matrix-hermitian"),
            ("/future/positions", "six-matrix-response.frame.six-matrix-hermitian"),
            ("/history/alpha_tilde", "six-matrix-response.frame.xy-coupling"),
            ("/history/innovations", "six-matrix-response.frame.six-matrix-hermitian"),
            ("/history/momenta", "six-matrix-response.frame.six-matrix-hermitian"),
            ("/history/positions", "six-matrix-response.frame.six-matrix-hermitian"),
            ("/history/step_index", "six-matrix-response.frame.integration-index"),
            ("/routing/alpha_tilde", "six-matrix-response.frame.xy-coupling"),
            ("/routing/momenta", "six-matrix-response.frame.six-matrix-hermitian"),
            ("/routing/positions", "six-matrix-response.frame.six-matrix-hermitian"),
        )
    )
    + "}"
)
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_CLOCKS = (
    "{"
    + ",".join(
        f'"{path}":"six-matrix-response.clock.native-integration-step"'
        for path in (
            "/future/arm",
            "/future/checkpoint_step",
            "/future/future_index",
            "/future/innovations",
            "/future/momenta",
            "/future/positions",
            "/history/alpha_tilde",
            "/history/innovations",
            "/history/momenta",
            "/history/positions",
            "/history/step_index",
            "/routing/alpha_tilde",
            "/routing/momenta",
            "/routing/positions",
        )
    )
    + "}"
)
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PRIMARY_MEMBER_ID = "matrix-response-prospective-reactive-source-law.primary"
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FINE_MEMBER_ID = "matrix-response-prospective-reactive-source-law.same-driver-dt-half"
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HISTORY_COUNT = 256
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FINE_COUNT = 64
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PREPARATION_STEPS = 1024
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PREPARATION_RAMP_STEPS = 256
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HOLD_STEPS = 4096
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FUTURE_STEPS = 512
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS = tuple(range(512, 4097, 64))
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_DEVELOPMENT_ROUTING_CHECKPOINTS = 16
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_DEVELOPMENT_ROUTING_FUTURES = 4
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PROSPECTIVE_SOURCE_FUTURES = 16
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PROSPECTIVE_CONTROL_FUTURES = 8
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FAMILIES = FOUR_FAMILY_PREPARATION_IDS
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_X = Decimal("0.6666666666666666666666666667")
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_Y = Decimal("7.333333333333333333333333334")
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HISTORY_SEED_RULE = "matrix-response-prospective-reactive-source-law.history-pcg64dxsm"
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_BRIDGE_SEED_RULE = "matrix-response-prospective-reactive-source-law.bridge-pcg64dxsm"
SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FUTURE_SEED_RULE = "matrix-response-prospective-reactive-source-law.future-after-routing-pcg64dxsm"
_MATRIX_SHAPE = (2, 3, 4, 4)


class SixMatrixResponseProspectiveReactiveSourceLawTranche(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    PROSPECTIVE_EVALUATION = "PROSPECTIVE_EVALUATION"


class SixMatrixResponseProspectiveReactiveSourceLawHalf(StrEnum):
    FIRST = "FIRST"
    SECOND = "SECOND"


class SixMatrixResponseProspectiveReactiveSourceLawNumericalView(StrEnum):
    PRIMARY = "PRIMARY"
    SAME_DRIVER_HALF = "SAME_DRIVER_HALF"


class SixMatrixResponseProspectiveReactiveSourceLawArm(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    SOURCE = "SOURCE"
    CONTROL = "CONTROL"


class SixMatrixResponseProspectiveReactiveSourceLawHistoryTerminal(StrEnum):
    COMPLETE = "COMPLETE"
    NUMERICAL_INVALID = "NUMERICAL_INVALID"


@dataclass(frozen=True, slots=True)
class SixMatrixResponseProspectiveReactiveSourceLawHistorySlot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-prospective-reactive-source-law-history-slot'

    slot_id: str
    history_index: int
    family_id: str
    half: SixMatrixResponseProspectiveReactiveSourceLawHalf
    physical_independent_unit_id: str
    preparation_instance_id: str
    primary_acquisition_group_id: str
    primary_view_id: str
    fine_acquisition_group_id: str | None
    fine_view_id: str | None

    def __post_init__(self) -> None:
        for name in (
            "slot_id",
            "family_id",
            "physical_independent_unit_id",
            "preparation_instance_id",
            "primary_acquisition_group_id",
            "primary_view_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if not 0 <= self.history_index < SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HISTORY_COUNT:
            raise ValueError("matrix response prospective reactive source law history index differs")
        if self.family_id != SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FAMILIES[self.history_index % 4]:
            raise ValueError("matrix response prospective reactive source law family allocation differs")
        expected_half = SixMatrixResponseProspectiveReactiveSourceLawHalf.FIRST if self.history_index < 128 else SixMatrixResponseProspectiveReactiveSourceLawHalf.SECOND
        if self.half is not expected_half:
            raise ValueError("matrix response prospective reactive source law half allocation differs")
        if (self.fine_acquisition_group_id is None) != (self.fine_view_id is None):
            raise ValueError("matrix response prospective reactive source law fine view identity is incomplete")
        if self.fine_acquisition_group_id is not None:
            validate_stable_id(self.fine_acquisition_group_id, field_name="fine_group")
            validate_stable_id(self.fine_view_id or "", field_name="fine_view")

    @property
    def has_fine_view(self) -> bool:
        return self.fine_view_id is not None


def build_six_matrix_response_reactive_source_slots(tranche: SixMatrixResponseProspectiveReactiveSourceLawTranche) -> tuple[SixMatrixResponseProspectiveReactiveSourceLawHistorySlot, ...]:
    """Freeze balanced halves/families and 16 fine histories per family."""

    scientific_inputs = reactive_source_history_scientific_inputs(
        tranche_ordinal=0 if tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.DEVELOPMENT else 1
    )
    fine = {value.history_index for value in scientific_inputs if value.has_fine_view}
    slots = []
    for index in range(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HISTORY_COUNT):
        stem = f"matrix-response-prospective-reactive-source-law.{tranche.value.lower()}.h{index:03d}"
        slots.append(
            SixMatrixResponseProspectiveReactiveSourceLawHistorySlot(
                slot_id=f"slot.{stem}",
                history_index=index,
                family_id=SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FAMILIES[index % 4],
                half=SixMatrixResponseProspectiveReactiveSourceLawHalf.FIRST if index < 128 else SixMatrixResponseProspectiveReactiveSourceLawHalf.SECOND,
                physical_independent_unit_id=f"unit.{stem}",
                preparation_instance_id=f"preparation.{stem}",
                primary_acquisition_group_id=f"acquisition.{stem}.primary",
                primary_view_id=f"view.{stem}.primary",
                fine_acquisition_group_id=f"acquisition.{stem}.fine" if index in fine else None,
                fine_view_id=f"view.{stem}.fine" if index in fine else None,
            )
        )
    return tuple(slots)


@dataclass(frozen=True, slots=True)
class SixMatrixResponseProspectiveReactiveSourceLawSourceConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-prospective-reactive-source-law-source-config'

    config_id: str
    config_version: str
    tranche: SixMatrixResponseProspectiveReactiveSourceLawTranche
    scientific_fingerprint: str
    seed_namespace_id: str
    member: SixMatrixResponseModelFamilyMember
    primary_view: SixMatrixResponseNumericalView
    fine_view: SixMatrixResponseNumericalView
    slots: tuple[SixMatrixResponseProspectiveReactiveSourceLawHistorySlot, ...]
    scientific_history_inputs: tuple[MatrixReactiveSourceHistoryScientificInput, ...]
    routing_scientific_inputs: tuple[MatrixReactiveSourceRoutingScientificInput, ...]
    original_routing_census_sha256: str
    frozen_score: ObjectIdentity | None
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.seed_namespace_id, field_name="seed_namespace_id")
        validate_semantic_version(self.config_version)
        validate_sha256(self.scientific_fingerprint, field_name="scientific_fingerprint")
        validate_sha256(self.original_routing_census_sha256, field_name="original_routing_census_sha256")
        require_sorted_unique_ids(self.slots, attribute="slot_id", field_name="slots")
        if (
            self.member.member_id != "six-matrix-response.member.mass-0p5.cross-coupling-1"
            or self.primary_view.timestep != Decimal("0.001")
            or self.fine_view.timestep != Decimal("0.0005")
            or self.primary_view.friction_gamma != self.fine_view.friction_gamma
            or self.primary_view.bath_temperature != self.fine_view.bath_temperature
            or self.slots != build_six_matrix_response_reactive_source_slots(self.tranche)
            or (self.frozen_score is None) != (self.tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.DEVELOPMENT)
            or self.grants_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("matrix response prospective reactive source law source contract differs")
        expected_history_inputs = reactive_source_history_scientific_inputs(
            tranche_ordinal=0 if self.tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.DEVELOPMENT else 1
        )
        if self.scientific_history_inputs != expected_history_inputs:
            raise ValueError("reactive-source history scientific input census differs")
        expected_routes = tuple(
            (slot.history_index, view_ordinal)
            for slot in self.slots
            for view_ordinal in (0, 1)
            if not view_ordinal or slot.has_fine_view
        )
        if (
            any(not isinstance(value, MatrixReactiveSourceRoutingScientificInput) for value in self.routing_scientific_inputs)
            or tuple((value.history_index, value.view_ordinal) for value in self.routing_scientific_inputs) != expected_routes
        ):
            raise ValueError("reactive-source requires a complete issued routing scientific input census")
        if original_routing_scientific_input_sha256(self.routing_scientific_inputs) != self.original_routing_census_sha256:
            raise ValueError("reactive-source original routing scientific input custody differs")
        if self.tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.DEVELOPMENT and self.original_routing_census_sha256 != '3a31a0fc9fb18aa98b403f261a0f43597f6db356dd9b2688777b431b35a7db42':
            raise ValueError("reactive-source development scientific input differs from the frozen original census")
        for value in self.routing_scientific_inputs:
            if self.tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.DEVELOPMENT:
                routing = development_routing(self, self.slot(value.history_index),
                    SixMatrixResponseProspectiveReactiveSourceLawNumericalView.PRIMARY if value.view_ordinal == 0 else SixMatrixResponseProspectiveReactiveSourceLawNumericalView.SAME_DRIVER_HALF, ())
                value.require_target_routing(
                    target_routing_sha256=routing.fingerprint(),
                    branch_coordinates=tuple(_branch_scientific_coordinates(spec) for spec in routing.branches),
                )
            else:
                coordinates = tuple(item.coordinates for item in value.futures)
                if any(arm not in (0, 2) for arm, _, _ in coordinates):
                    raise ValueError("reactive-source evaluation future arm input differs")
                expected = []
                for arm in (0, 2):
                    steps = {step for actual_arm, step, _ in coordinates if actual_arm == arm}
                    if len(steps) > 1:
                        raise ValueError("reactive-source evaluation requires one checkpoint per arm")
                    for step in sorted(steps):
                        count = 1 if value.view_ordinal else (8 if arm == 0 else 16)
                        expected.extend((arm, step, index) for index in range(count))
                if tuple(expected) != coordinates:
                    raise ValueError("reactive-source evaluation future input census is incomplete")

    def slot(self, index: int) -> SixMatrixResponseProspectiveReactiveSourceLawHistorySlot:
        return self.slots[index]

    def routing_scientific_input(self, history_index: int, numerical_view: SixMatrixResponseProspectiveReactiveSourceLawNumericalView) -> MatrixReactiveSourceRoutingScientificInput:
        view_ordinal = 0 if numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.PRIMARY else 1
        for value in self.routing_scientific_inputs:
            if (value.history_index, value.view_ordinal) == (history_index, view_ordinal):
                return value
        raise ValueError("reactive-source routing lacks an issued scientific input")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseProspectiveReactiveSourceLawBranchSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-prospective-reactive-source-law-branch-spec'

    branch_id: str
    arm: SixMatrixResponseProspectiveReactiveSourceLawArm
    checkpoint_step: int
    future_index: int

    def __post_init__(self) -> None:
        validate_stable_id(self.branch_id, field_name="branch_id")
        if self.checkpoint_step not in SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS or self.future_index < 0:
            raise ValueError("matrix response prospective reactive source law branch coordinate differs")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseProspectiveReactiveSourceLawRouting(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-prospective-reactive-source-law-routing'

    routing_id: str
    tranche: SixMatrixResponseProspectiveReactiveSourceLawTranche
    history_index: int
    numerical_view: SixMatrixResponseProspectiveReactiveSourceLawNumericalView
    frozen_score: ObjectIdentity | None
    eligible_steps: tuple[int, ...]
    selected_development_steps: tuple[int, ...]
    source_step: int | None
    control_step: int | None
    branches: tuple[SixMatrixResponseProspectiveReactiveSourceLawBranchSpec, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.routing_id, field_name="routing_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if tuple(sorted(set(self.eligible_steps))) != self.eligible_steps or any(
            value not in SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS for value in self.eligible_steps
        ):
            raise ValueError("matrix response prospective reactive source law eligible-step roster differs")
        if tuple(sorted(set(self.selected_development_steps))) != self.selected_development_steps:
            raise ValueError("matrix response prospective reactive source law development selection differs")
        fine = self.numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.SAME_DRIVER_HALF
        if self.tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.DEVELOPMENT:
            expected_per_checkpoint = 1 if fine else SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_DEVELOPMENT_ROUTING_FUTURES
            if (
                self.frozen_score is not None
                or len(self.selected_development_steps) != SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_DEVELOPMENT_ROUTING_CHECKPOINTS
                or self.source_step is not None
                or self.control_step is not None
                or len(self.branches) != SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_DEVELOPMENT_ROUTING_CHECKPOINTS * expected_per_checkpoint
                or any(value.arm is not SixMatrixResponseProspectiveReactiveSourceLawArm.DEVELOPMENT for value in self.branches)
            ):
                raise ValueError("matrix response prospective reactive source law DEVELOPMENT routing differs")
        else:
            if self.frozen_score is None or self.selected_development_steps:
                raise ValueError("matrix response prospective reactive source law PROSPECTIVE_EVALUATION routing lacks its frozen score")
            expected = (
                (int(self.source_step is not None) + int(self.control_step is not None))
                if fine
                else (
                    (SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PROSPECTIVE_SOURCE_FUTURES if self.source_step is not None else 0)
                    + (SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PROSPECTIVE_CONTROL_FUTURES if self.control_step is not None else 0)
                )
            )
            if len(self.branches) != expected:
                raise ValueError("matrix response prospective reactive source law PROSPECTIVE_EVALUATION future roster differs")
        coordinates = tuple(
            (value.arm.value, value.checkpoint_step, value.future_index) for value in self.branches
        )
        if tuple(sorted(coordinates)) != coordinates or len(set(coordinates)) != len(coordinates):
            raise ValueError("matrix response prospective reactive source law branches are not canonical")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseProspectiveReactiveSourceLawHistoryRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-prospective-reactive-source-law-history-request'

    request_id: str
    task_id: str
    result_output_id: str
    trajectory_output_id: str
    source_config: ObjectIdentity
    neutral_request: NativeAcquisitionRequest
    slot: SixMatrixResponseProspectiveReactiveSourceLawHistorySlot
    numerical_view: SixMatrixResponseProspectiveReactiveSourceLawNumericalView
    acquisition_group_id: str
    scientific_view_id: str
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "task_id",
            "result_output_id",
            "trajectory_output_id",
            "acquisition_group_id",
            "scientific_view_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        primary = self.numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.PRIMARY
        if (
            self.acquisition_group_id
            != (
                self.slot.primary_acquisition_group_id
                if primary
                else self.slot.fine_acquisition_group_id
            )
            or self.scientific_view_id
            != (self.slot.primary_view_id if primary else self.slot.fine_view_id)
            or self.neutral_request.physical_independent_unit_id
            != self.slot.physical_independent_unit_id
            or self.neutral_request.acquisition_group_id != self.acquisition_group_id
            or self.evidence_ceiling is not EvidenceCeiling.MEASUREMENT
            or self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or self.grants_authority
        ):
            raise ValueError("matrix response prospective reactive source law history request differs")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseProspectiveReactiveSourceLawHistoryResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-prospective-reactive-source-law-history-result'

    result_id: str
    task_id: str
    request: ObjectIdentity
    neutral_request: ObjectIdentity
    source_config: ObjectIdentity
    slot_id: str
    history_index: int
    family_id: str
    physical_independent_unit_id: str
    acquisition_group_id: str
    scientific_view_id: str
    numerical_view: SixMatrixResponseProspectiveReactiveSourceLawNumericalView
    routing: SixMatrixResponseProspectiveReactiveSourceLawRouting
    routing_sha256: str
    history_state_count: int
    branch_count: int
    terminal: SixMatrixResponseProspectiveReactiveSourceLawHistoryTerminal
    hdf5_payload_schema: str
    hdf5_physical_sha256: str
    hdf5_size_bytes: int
    reason_codes: tuple[str, ...]
    scientific_verdict_constructed: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name in (
            "result_id",
            "task_id",
            "slot_id",
            "family_id",
            "physical_independent_unit_id",
            "acquisition_group_id",
            "scientific_view_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("routing_sha256", "hdf5_physical_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.routing_sha256 != self.routing.fingerprint()
            or self.history_state_count
            != 1
            + (SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PREPARATION_STEPS + SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HOLD_STEPS)
            * (1 if self.numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.PRIMARY else 2)
            or self.branch_count != len(self.routing.branches)
            or self.hdf5_payload_schema != SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA
            or self.hdf5_size_bytes <= 0
            or self.scientific_verdict_constructed
            or self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or (self.terminal is SixMatrixResponseProspectiveReactiveSourceLawHistoryTerminal.COMPLETE) != (not self.reason_codes)
        ):
            raise ValueError("matrix response prospective reactive source law history result differs")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseProspectiveReactiveSourceLawPersistedHistory:
    result: SixMatrixResponseProspectiveReactiveSourceLawHistoryResult
    history_states: tuple[SixMatrixState, ...]
    history_innovations: ComplexArray
    routing_states: tuple[SixMatrixState, ...]
    branch_states: tuple[tuple[SixMatrixState, ...], ...]
    branch_innovations: ComplexArray


ProspectiveReactiveSourceLawRoutingService = Callable[
    [
        SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
        SixMatrixResponseProspectiveReactiveSourceLawHistorySlot,
        SixMatrixResponseProspectiveReactiveSourceLawNumericalView,
        tuple[SixMatrixState, ...],
    ],
    SixMatrixResponseProspectiveReactiveSourceLawRouting,
]


def _scientific_rng(scientific_seed_sha256: str) -> tuple[np.random.Generator, str]:
    validate_sha256(scientific_seed_sha256, field_name="scientific_seed_sha256")
    digest = bytes.fromhex(scientific_seed_sha256)
    return np.random.Generator(
        np.random.PCG64DXSM(int.from_bytes(digest[:16], "big"))
    ), scientific_seed_sha256


def _branch_scientific_coordinates(spec: SixMatrixResponseProspectiveReactiveSourceLawBranchSpec) -> tuple[int, int, int]:
    # Numeric order preserves the original CONTROL, DEVELOPMENT, SOURCE order.
    arm_ordinal = (SixMatrixResponseProspectiveReactiveSourceLawArm.CONTROL, SixMatrixResponseProspectiveReactiveSourceLawArm.DEVELOPMENT, SixMatrixResponseProspectiveReactiveSourceLawArm.SOURCE).index(spec.arm)
    return arm_ordinal, spec.checkpoint_step, spec.future_index


def development_routing(
    config: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
    slot: SixMatrixResponseProspectiveReactiveSourceLawHistorySlot,
    numerical_view: SixMatrixResponseProspectiveReactiveSourceLawNumericalView,
    states: tuple[SixMatrixState, ...],
) -> SixMatrixResponseProspectiveReactiveSourceLawRouting:
    """Pre-outcome hash panel; state bytes are deliberately ignored."""

    del states
    selected = config.scientific_history_inputs[slot.history_index].development_checkpoint_steps
    if len(selected) != SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_DEVELOPMENT_ROUTING_CHECKPOINTS:
        raise ValueError("reactive-source development routing lacks its issued numerical order")
    count = 1 if numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.SAME_DRIVER_HALF else SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_DEVELOPMENT_ROUTING_FUTURES
    branches = tuple(
        sorted(
            (
                SixMatrixResponseProspectiveReactiveSourceLawBranchSpec(
                    branch_id=f"branch.development.h{slot.history_index:03d}.c{step:04d}.f{future:02d}",
                    arm=SixMatrixResponseProspectiveReactiveSourceLawArm.DEVELOPMENT,
                    checkpoint_step=step,
                    future_index=future,
                )
                for step in selected
                for future in range(count)
            ),
            key=lambda value: (value.arm.value, value.checkpoint_step, value.future_index),
        )
    )
    return SixMatrixResponseProspectiveReactiveSourceLawRouting(
        routing_id=f"routing.development.h{slot.history_index:03d}.{numerical_view.value.lower()}",
        tranche=config.tranche,
        history_index=slot.history_index,
        numerical_view=numerical_view,
        frozen_score=None,
        eligible_steps=SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS,
        selected_development_steps=selected,
        source_step=None,
        control_step=None,
        branches=branches,
        reason_codes=(),
    )


def _noise_block(rng: np.random.Generator, count: int) -> ComplexArray:
    return np.ascontiguousarray(
        np.stack([hermitian_noise(rng=rng, q=2) for _ in range(count)]), dtype="<c16"
    )


def _advance(
    initial: SixMatrixState,
    noises: ComplexArray,
    *,
    config: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
    view: SixMatrixResponseNumericalView,
    coupling: Callable[[int], tuple[float, float]],
) -> tuple[SixMatrixState, ...]:
    gradient_cache_state = BAOABGradientCache()
    state = initial
    states = [state]
    for index, noise in enumerate(noises, 1):
        x, y = coupling(index)
        state = baoab_step_with_hermitian_noise(
            state,
            member=config.member,
            numerical_view=view,
            next_alpha_tilde_x=x,
            next_alpha_tilde_y=y,
            standardized_noise=noise,
            gradient_cache=gradient_cache_state,
        )
        if not state.finite or hermiticity_residual(state.positions) > 1e-10:
            raise FloatingPointError("matrix response prospective reactive source law integration became invalid")
        states.append(state)
    return tuple(states)


def _history(
    config: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
    slot: SixMatrixResponseProspectiveReactiveSourceLawHistorySlot,
    numerical_view: SixMatrixResponseProspectiveReactiveSourceLawNumericalView,
) -> tuple[tuple[SixMatrixState, ...], ComplexArray, tuple[SixMatrixState, ...]]:
    coarse_rng, _ = _scientific_rng(config.scientific_history_inputs[slot.history_index].history_seed_sha256)
    coarse_count = SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PREPARATION_STEPS + SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HOLD_STEPS
    coarse = _noise_block(coarse_rng, coarse_count)
    initial = ideal_state(
        q=2,
        alpha_tilde_x=8.0 if slot.family_id == "matrix-history.joint-decreasing-coupling" else 0.0,
        alpha_tilde_y=8.0 if slot.family_id == "matrix-history.joint-decreasing-coupling" else 0.0,
        constitution="11" if slot.family_id == "matrix-history.joint-decreasing-coupling" else "00",
    )
    primary_states = _advance(
        initial,
        coarse,
        config=config,
        view=config.primary_view,
        coupling=lambda step: (
            four_family_preparation_couplings(
                slot.family_id,
                step,
                ramp_steps=SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PREPARATION_RAMP_STEPS,
                integrator_multiplier=1,
                target_x=float(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_X),
                target_y=float(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_Y),
            )
            if step <= SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PREPARATION_STEPS
            else (float(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_X), float(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_Y))
        ),
    )
    if numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.PRIMARY:
        noises, multiplier, view = coarse, 1, config.primary_view
    else:
        bridge_rng, _ = _scientific_rng(config.scientific_history_inputs[slot.history_index].bridge_seed_sha256)
        bridge = _noise_block(bridge_rng, coarse_count)
        decay = float(np.exp(-float(config.primary_view.friction_gamma) * 0.001 / 2.0))
        noises = np.ascontiguousarray(
            np.stack(
                [
                    item
                    for left, right in zip(coarse, bridge, strict=True)
                    for item in brownian_bridge_split(
                        coarse_noise=left, bridge_noise=right, half_decay=decay
                    )
                ]
            ),
            dtype="<c16",
        )
        multiplier, view = 2, config.fine_view
    prep = SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PREPARATION_STEPS * multiplier
    states = _advance(
        initial,
        noises,
        config=config,
        view=view,
        coupling=lambda step: (
            four_family_preparation_couplings(
                slot.family_id,
                step,
                ramp_steps=SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PREPARATION_RAMP_STEPS,
                integrator_multiplier=multiplier,
                target_x=float(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_X),
                target_y=float(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_Y),
            )
            if step <= prep
            else (float(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_X), float(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_Y))
        ),
    )
    return states, noises, primary_states


def _future(
    *,
    config: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
    slot: SixMatrixResponseProspectiveReactiveSourceLawHistorySlot,
    numerical_view: SixMatrixResponseProspectiveReactiveSourceLawNumericalView,
    routing_sha256: str,
    spec: SixMatrixResponseProspectiveReactiveSourceLawBranchSpec,
    initial: SixMatrixState,
) -> tuple[tuple[SixMatrixState, ...], ComplexArray]:
    routing_input = config.routing_scientific_input(slot.history_index, numerical_view)
    if routing_sha256 != routing_input.target_routing_sha256:
        raise ValueError("reactive-source future target routing custody differs")
    scientific_input = routing_input.future(_branch_scientific_coordinates(spec))
    coarse_rng, _ = _scientific_rng(scientific_input.scientific_seed_sha256)
    coarse = _noise_block(coarse_rng, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FUTURE_STEPS)
    if numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.PRIMARY:
        noises, view = coarse, config.primary_view
    else:
        if scientific_input.bridge_seed_sha256 is None:
            raise ValueError("reactive-source fine future lacks an issued bridge seed input")
        bridge_rng, _ = _scientific_rng(scientific_input.bridge_seed_sha256)
        bridge = _noise_block(bridge_rng, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FUTURE_STEPS)
        decay = float(np.exp(-float(config.primary_view.friction_gamma) * 0.001 / 2.0))
        noises = np.ascontiguousarray(
            np.stack(
                [
                    item
                    for left, right in zip(coarse, bridge, strict=True)
                    for item in brownian_bridge_split(
                        coarse_noise=left, bridge_noise=right, half_decay=decay
                    )
                ]
            ),
            dtype="<c16",
        )
        view = config.fine_view
    reset = SixMatrixState(
        q=initial.q,
        positions=initial.positions,
        momenta=initial.momenta,
        step_index=0,
        alpha_tilde_x=initial.alpha_tilde_x,
        alpha_tilde_y=initial.alpha_tilde_y,
    )
    states = _advance(
        reset,
        noises,
        config=config,
        view=view,
        coupling=lambda _: (float(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_X), float(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_Y)),
    )
    return states, noises


def _h5py() -> Any:
    return importlib.import_module("h5py")


def _hdf5(
    *,
    config: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
    request: SixMatrixResponseProspectiveReactiveSourceLawHistoryRequest,
    routing: SixMatrixResponseProspectiveReactiveSourceLawRouting,
    history_states: tuple[SixMatrixState, ...],
    history_noise: ComplexArray,
    routing_states: tuple[SixMatrixState, ...],
    branch_states: tuple[tuple[SixMatrixState, ...], ...],
    branch_noise: ComplexArray,
) -> bytes:
    stream = BytesIO()
    h5py = _h5py()
    positions = np.ascontiguousarray(
        np.stack([value.positions for value in history_states]), dtype="<c16"
    )
    momenta = np.ascontiguousarray(
        np.stack([value.momenta for value in history_states]), dtype="<c16"
    )
    alpha = np.asarray(
        [(value.alpha_tilde_x, value.alpha_tilde_y) for value in history_states], dtype="<f8"
    )
    routing_positions = np.ascontiguousarray(
        np.stack([value.positions for value in routing_states]), dtype="<c16"
    )
    routing_momenta = np.ascontiguousarray(
        np.stack([value.momenta for value in routing_states]), dtype="<c16"
    )
    routing_alpha = np.asarray(
        [(value.alpha_tilde_x, value.alpha_tilde_y) for value in routing_states], dtype="<f8"
    )
    branch_positions = np.ascontiguousarray(
        np.stack([[value.positions for value in path] for path in branch_states])
        if branch_states
        else np.empty(
            (
                0,
                1
                + SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FUTURE_STEPS
                * (1 if request.numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.PRIMARY else 2),
                *_MATRIX_SHAPE,
            ),
            dtype="<c16",
        ),
        dtype="<c16",
    )
    branch_momenta = np.ascontiguousarray(
        np.stack([[value.momenta for value in path] for path in branch_states])
        if branch_states
        else np.empty_like(branch_positions),
        dtype="<c16",
    )
    with h5py.File(stream, "w", libver="earliest", track_order=False) as handle:
        attrs = {
            "empirical_lawhood_clocks": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_CLOCKS,
            "empirical_lawhood_frames": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_FRAMES,
            "empirical_lawhood_keys": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_KEYS,
            "empirical_lawhood_payload_schema": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA,
            "empirical_lawhood_units": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_UNITS,
            "source_config_fingerprint": config.fingerprint(),
            "request_fingerprint": request.fingerprint(),
            "routing_fingerprint": routing.fingerprint(),
            "physical_independent_unit_id": request.slot.physical_independent_unit_id,
            "compression": "gzip-1",
        }
        for key, value in sorted(attrs.items()):
            handle.attrs.create(key, str(value).encode(), dtype=f"S{len(str(value).encode())}")
        history = handle.create_group("history", track_order=False)
        routing_group = handle.create_group("routing", track_order=False)
        future = handle.create_group("future", track_order=False)
        kwargs = {
            "compression": "gzip",
            "compression_opts": 1,
            "shuffle": False,
            "track_times": False,
        }
        history.create_dataset("positions", data=positions, **kwargs)
        history.create_dataset("momenta", data=momenta, **kwargs)
        history.create_dataset("alpha_tilde", data=alpha, **kwargs)
        history.create_dataset(
            "innovations", data=np.asarray(history_noise, dtype="<c16"), **kwargs
        )
        history.create_dataset(
            "step_index", data=np.arange(len(history_states), dtype="<i8"), track_times=False
        )
        routing_group.create_dataset("positions", data=routing_positions, **kwargs)
        routing_group.create_dataset("momenta", data=routing_momenta, **kwargs)
        routing_group.create_dataset("alpha_tilde", data=routing_alpha, **kwargs)
        future.create_dataset("positions", data=branch_positions, **kwargs)
        future.create_dataset("momenta", data=branch_momenta, **kwargs)
        future.create_dataset("innovations", data=np.asarray(branch_noise, dtype="<c16"), **kwargs)
        future.create_dataset(
            "checkpoint_step",
            data=np.asarray([value.checkpoint_step for value in routing.branches], dtype="<i8"),
            track_times=False,
        )
        future.create_dataset(
            "future_index",
            data=np.asarray([value.future_index for value in routing.branches], dtype="<i8"),
            track_times=False,
        )
        arm = np.asarray([value.arm.value.encode() for value in routing.branches], dtype="S11")
        future.create_dataset("arm", data=arm, track_times=False)
    return stream.getvalue()


def execute_six_matrix_response_reactive_source_history(
    *,
    config: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
    request: SixMatrixResponseProspectiveReactiveSourceLawHistoryRequest,
    routing_service: ProspectiveReactiveSourceLawRoutingService | None,
) -> tuple[SixMatrixResponseProspectiveReactiveSourceLawHistoryResult, bytes]:
    """Execute history, freeze routing, then derive and execute future streams."""

    if request.source_config != ObjectIdentity.from_record(config.config_id, config):
        raise ValueError("matrix response prospective reactive source law request changes its source")
    if request.slot != config.slot(request.slot.history_index):
        raise ValueError("reactive-source request changes its issued history slot")
    routing_input = config.routing_scientific_input(request.slot.history_index, request.numerical_view)
    reasons: set[str] = set()
    try:
        states, innovations, routing_states = _history(config, request.slot, request.numerical_view)
        if config.tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.DEVELOPMENT:
            routing = development_routing(config, request.slot, request.numerical_view, routing_states)
        else:
            if routing_service is None:
                raise ValueError("matrix response prospective reactive source law PROSPECTIVE_EVALUATION lacks its frozen routing service")
            routing = routing_service(config, request.slot, request.numerical_view, routing_states)
        routing_sha256 = routing.fingerprint()
        routing_input.require_target_routing(
            target_routing_sha256=routing_sha256,
            branch_coordinates=tuple(_branch_scientific_coordinates(spec) for spec in routing.branches),
        )
        multiplier = 1 if request.numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.PRIMARY else 2
        prep = SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PREPARATION_STEPS * multiplier
        branch_values = tuple(
            _future(
                config=config,
                slot=request.slot,
                numerical_view=request.numerical_view,
                routing_sha256=routing_sha256,
                spec=spec,
                initial=states[prep + spec.checkpoint_step * multiplier],
            )
            for spec in routing.branches
        )
        branches = tuple(value[0] for value in branch_values)
        branch_noise = np.ascontiguousarray(
            np.stack([value[1] for value in branch_values])
            if branch_values
            else np.empty((0, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FUTURE_STEPS * multiplier, *_MATRIX_SHAPE), dtype="<c16"),
            dtype="<c16",
        )
    except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError) as error:
        reasons.add(f"numerical-or-routing-invalid:{type(error).__name__}")
        raise
    payload = _hdf5(
        config=config,
        request=request,
        routing=routing,
        history_states=states,
        history_noise=innovations,
        routing_states=routing_states,
        branch_states=branches,
        branch_noise=branch_noise,
    )
    result = SixMatrixResponseProspectiveReactiveSourceLawHistoryResult(
        result_id=f"history-result.{request.task_id}",
        task_id=request.task_id,
        request=ObjectIdentity.from_record(request.request_id, request),
        neutral_request=ObjectIdentity.from_record(
            request.neutral_request.request_id, request.neutral_request
        ),
        source_config=ObjectIdentity.from_record(config.config_id, config),
        slot_id=request.slot.slot_id,
        history_index=request.slot.history_index,
        family_id=request.slot.family_id,
        physical_independent_unit_id=request.slot.physical_independent_unit_id,
        acquisition_group_id=request.acquisition_group_id,
        scientific_view_id=request.scientific_view_id,
        numerical_view=request.numerical_view,
        routing=routing,
        routing_sha256=routing.fingerprint(),
        history_state_count=len(states),
        branch_count=len(branches),
        terminal=SixMatrixResponseProspectiveReactiveSourceLawHistoryTerminal.COMPLETE,
        hdf5_payload_schema=SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA,
        hdf5_physical_sha256=sha256(payload).hexdigest(),
        hdf5_size_bytes=len(payload),
        reason_codes=tuple(sorted(reasons)),
        scientific_verdict_constructed=False,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    return result, payload


def decode_six_matrix_response_reactive_source_hdf5(
    payload: bytes, result: SixMatrixResponseProspectiveReactiveSourceLawHistoryResult
) -> SixMatrixResponseProspectiveReactiveSourceLawPersistedHistory:
    """Strictly reconstruct the complete history and future panel."""

    if (
        len(payload) != result.hdf5_size_bytes
        or sha256(payload).hexdigest() != result.hdf5_physical_sha256
    ):
        raise ValueError("matrix response prospective reactive source law HDF5 physical identity differs")
    h5py = _h5py()
    with h5py.File(BytesIO(payload), "r") as handle:
        observed: set[str] = set()
        handle.visit(observed.add)
        expected = {
            "future",
            "future/arm",
            "future/checkpoint_step",
            "future/future_index",
            "future/innovations",
            "future/momenta",
            "future/positions",
            "history",
            "history/alpha_tilde",
            "history/innovations",
            "history/momenta",
            "history/positions",
            "history/step_index",
            "routing",
            "routing/alpha_tilde",
            "routing/momenta",
            "routing/positions",
        }
        if observed != expected:
            raise ValueError("matrix response prospective reactive source law HDF5 inventory differs")
        attrs = {key: bytes(value).rstrip(b"\0").decode() for key, value in handle.attrs.items()}
        if (
            attrs.get("empirical_lawhood_payload_schema") != SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA
            or attrs.get("routing_fingerprint") != result.routing_sha256
        ):
            raise ValueError("matrix response prospective reactive source law HDF5 custody attributes differ")
        hp = np.asarray(handle["history/positions"], dtype="<c16")
        hm = np.asarray(handle["history/momenta"], dtype="<c16")
        ha = np.asarray(handle["history/alpha_tilde"], dtype="<f8")
        hn = np.asarray(handle["history/innovations"], dtype="<c16")
        hs = np.asarray(handle["history/step_index"], dtype="<i8")
        rp = np.asarray(handle["routing/positions"], dtype="<c16")
        rm = np.asarray(handle["routing/momenta"], dtype="<c16")
        ra = np.asarray(handle["routing/alpha_tilde"], dtype="<f8")
        bp = np.asarray(handle["future/positions"], dtype="<c16")
        bm = np.asarray(handle["future/momenta"], dtype="<c16")
        bn = np.asarray(handle["future/innovations"], dtype="<c16")
        checkpoints = tuple(map(int, np.asarray(handle["future/checkpoint_step"], dtype="<i8")))
        futures = tuple(map(int, np.asarray(handle["future/future_index"], dtype="<i8")))
        arms = tuple(
            bytes(value).rstrip(b"\0").decode() for value in np.asarray(handle["future/arm"])
        )
    multiplier = 1 if result.numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.PRIMARY else 2
    if (
        hp.shape != (result.history_state_count, *_MATRIX_SHAPE)
        or hm.shape != hp.shape
        or ha.shape != (result.history_state_count, 2)
        or hn.shape != (result.history_state_count - 1, *_MATRIX_SHAPE)
        or tuple(map(int, hs)) != tuple(range(result.history_state_count))
        or rp.shape != (1 + SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PREPARATION_STEPS + SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HOLD_STEPS, *_MATRIX_SHAPE)
        or rm.shape != rp.shape
        or ra.shape != (rp.shape[0], 2)
        or bp.shape != (result.branch_count, 1 + SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FUTURE_STEPS * multiplier, *_MATRIX_SHAPE)
        or bm.shape != bp.shape
        or bn.shape != (result.branch_count, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FUTURE_STEPS * multiplier, *_MATRIX_SHAPE)
        or tuple((arms[i], checkpoints[i], futures[i]) for i in range(result.branch_count))
        != tuple(
            (value.arm.value, value.checkpoint_step, value.future_index)
            for value in result.routing.branches
        )
        or not all(np.isfinite(value).all() for value in (hp, hm, ha, hn, rp, rm, ra, bp, bm, bn))
        or hermiticity_residual(hp) > 1e-10
        or hermiticity_residual(hm) > 1e-10
    ):
        raise ValueError("matrix response prospective reactive source law HDF5 geometry differs")
    history_states = tuple(
        SixMatrixState(
            q=2,
            positions=hp[i],
            momenta=hm[i],
            step_index=i,
            alpha_tilde_x=float(ha[i, 0]),
            alpha_tilde_y=float(ha[i, 1]),
        )
        for i in range(result.history_state_count)
    )
    branch_states = tuple(
        tuple(
            SixMatrixState(
                q=2,
                positions=bp[j, i],
                momenta=bm[j, i],
                step_index=i,
                alpha_tilde_x=float(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_X),
                alpha_tilde_y=float(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_Y),
            )
            for i in range(bp.shape[1])
        )
        for j in range(result.branch_count)
    )
    routing_states = tuple(
        SixMatrixState(
            q=2,
            positions=rp[i],
            momenta=rm[i],
            step_index=i,
            alpha_tilde_x=float(ra[i, 0]),
            alpha_tilde_y=float(ra[i, 1]),
        )
        for i in range(rp.shape[0])
    )
    return SixMatrixResponseProspectiveReactiveSourceLawPersistedHistory(result, history_states, hn, routing_states, branch_states, bn)


def six_matrix_response_reactive_source_scientific_fingerprint() -> str:
    """Implementation-neutral digest of the frozen baseline design constants."""

    return sha256(
        canonical_json_bytes(
            {
                "families": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FAMILIES,
                "histories_per_tranche": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HISTORY_COUNT,
                "fine_histories": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FINE_COUNT,
                "preparation_steps": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PREPARATION_STEPS,
                "hold_steps": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HOLD_STEPS,
                "candidate_steps": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS,
                "development_checkpoints": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_DEVELOPMENT_ROUTING_CHECKPOINTS,
                "development_futures": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_DEVELOPMENT_ROUTING_FUTURES,
                "evaluation_source_futures": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PROSPECTIVE_SOURCE_FUTURES,
                "evaluation_control_futures": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PROSPECTIVE_CONTROL_FUTURES,
                "future_steps": SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FUTURE_STEPS,
                "target": (str(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_X), str(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_Y)),
                "score": "l2-logistic-lambda-1-no-interactions",
                "thresholds": tuple(f"{value / 100:.2f}" for value in range(5, 100, 5)),
                "terminal_order": (
                    "PREREQUISITE_NONATTEMPT",
                    "CUSTODY_INVALID",
                    "NUMERICAL_INVALID",
                    "SOURCE_EVALUATION_UNEVALUABLE",
                    "PROSPECTIVE_REACTIVE_SOURCE_SUPPORTED",
                    "CONDITIONAL_SOURCE_ONLY",
                    "NO_PROSPECTIVE_SOURCE",
                ),
                "top_up": False,
            }
        )
    ).hexdigest()


__all__ = [
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_MEDIA_TYPE',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANONICAL_MEDIA_TYPE',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_KEYS',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_UNITS',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_FRAMES',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_CLOCKS',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PRIMARY_MEMBER_ID',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FINE_MEMBER_ID',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HISTORY_COUNT',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FINE_COUNT',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PREPARATION_STEPS',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PREPARATION_RAMP_STEPS',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HOLD_STEPS',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FUTURE_STEPS',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_DEVELOPMENT_ROUTING_CHECKPOINTS',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_DEVELOPMENT_ROUTING_FUTURES',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PROSPECTIVE_SOURCE_FUTURES',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PROSPECTIVE_CONTROL_FUTURES',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FAMILIES',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_X',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_TARGET_Y',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HISTORY_SEED_RULE',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_BRIDGE_SEED_RULE',
    'SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FUTURE_SEED_RULE',
    'SixMatrixResponseProspectiveReactiveSourceLawTranche',
    'SixMatrixResponseProspectiveReactiveSourceLawHalf',
    'SixMatrixResponseProspectiveReactiveSourceLawNumericalView',
    'SixMatrixResponseProspectiveReactiveSourceLawArm',
    'SixMatrixResponseProspectiveReactiveSourceLawHistoryTerminal',
    'SixMatrixResponseProspectiveReactiveSourceLawHistorySlot',
    'build_six_matrix_response_reactive_source_slots',
    'SixMatrixResponseProspectiveReactiveSourceLawSourceConfig',
    'SixMatrixResponseProspectiveReactiveSourceLawBranchSpec',
    'SixMatrixResponseProspectiveReactiveSourceLawRouting',
    'SixMatrixResponseProspectiveReactiveSourceLawHistoryRequest',
    'SixMatrixResponseProspectiveReactiveSourceLawHistoryResult',
    'SixMatrixResponseProspectiveReactiveSourceLawPersistedHistory',
    'development_routing',
    'execute_six_matrix_response_reactive_source_history',
    'decode_six_matrix_response_reactive_source_hdf5',
    'six_matrix_response_reactive_source_scientific_fingerprint',
]
