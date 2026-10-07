"Pure source-evidence construction of qualified controlled-I/O members.\n\nThe substrate adapter owns acquisition and matrix extraction.  This module owns\nonly deterministic validation, normalization continuity, bounded numerical\ndiagnostics and construction of the existing :class:`ControlledIOMember`.\nIt performs no source access, persistence, law qualification or admission decision.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import CoordinateOrigin

from .contracts import (
    ControlledIOMember,
    ControlledIOMemberDiagnostics,
    ControlledIOProductDisposition,
    ControlledIOQualificationConfig,
    ControlledIOStep,
    CoordinateBasis,
    decimal_from_float,
)


MAX_CONTROLLED_IO_STATE_DIMENSION = 512
MAX_CONTROLLED_IO_INPUT_DIMENSION = 1
MAX_CONTROLLED_IO_RECEIVER_DIMENSION = 1
MAX_CONTROLLED_IO_STEPS = 8
MAX_OPERATOR_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_MATERIALIZED_MEMBER_BYTES = 256 * 1024 * 1024
MAX_OPERATOR_STAGE_BYTES = 16 * 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ControlledIOClockAxis(CanonicalRecord):
    """One exact source clock axis used by the controlled operator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-clock-axis'

    axis_id: str
    clock_id: str
    time_unit: str
    coordinate_frame: str
    origin: CoordinateOrigin

    def __post_init__(self) -> None:
        validate_stable_id(self.axis_id, field_name="axis_id")
        validate_stable_id(self.clock_id, field_name="clock_id")
        validate_nonempty(self.time_unit, field_name="time_unit")
        validate_nonempty(self.coordinate_frame, field_name="coordinate_frame")


@dataclass(frozen=True, slots=True)
class ControlledIOOperatorStepTopology(CanonicalRecord):
    """Exact source slot and native clock coordinates for one operator step."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-operator-step-topology'

    step_id: str
    step_index: int
    source_state_request_index: int
    source_input_request_index: int | None
    state_coordinate: Decimal
    input_coordinate: Decimal | None
    receiver_coordinate: Decimal
    terminal_closure: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.step_id, field_name="step_id")
        if self.step_index < 0 or self.source_state_request_index < 0:
            raise ValueError("controlled-I/O topology indices must be nonnegative")
        if self.source_input_request_index is not None and self.source_input_request_index < 0:
            raise ValueError("controlled-I/O input request index must be nonnegative")
        validate_decimal(self.state_coordinate, field_name="state_coordinate")
        validate_decimal(self.receiver_coordinate, field_name="receiver_coordinate")
        if self.input_coordinate is not None:
            validate_decimal(self.input_coordinate, field_name="input_coordinate")
        if self.terminal_closure != (self.source_input_request_index is None):
            raise ValueError("only the terminal controlled-I/O step may omit an input request")
        if self.terminal_closure != (self.input_coordinate is None):
            raise ValueError("only the terminal controlled-I/O step may omit an input coordinate")


@dataclass(frozen=True, slots=True)
class ControlledIOOperatorTopology(CanonicalRecord):
    """Complete predeclared source-slot and clock topology for one member family."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-operator-topology'

    topology_id: str
    state_clock: ControlledIOClockAxis
    input_clock: ControlledIOClockAxis
    receiver_clock: ControlledIOClockAxis
    steps: tuple[ControlledIOOperatorStepTopology, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.topology_id, field_name="topology_id")
        require_sorted_unique_ids(self.steps, attribute="step_id", field_name="steps")
        if not (2 <= len(self.steps) <= MAX_CONTROLLED_IO_STEPS):
            raise ValueError("controlled-I/O operator topology has an invalid step count")
        if tuple(value.step_index for value in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("controlled-I/O operator topology steps must be contiguous")
        state_requests = tuple(value.source_state_request_index for value in self.steps)
        if state_requests != tuple(
            range(state_requests[0], state_requests[0] + len(state_requests))
        ):
            raise ValueError("controlled-I/O source state requests must be contiguous")
        input_requests = tuple(value.source_input_request_index for value in self.steps[:-1])
        if (
            any(value is None for value in input_requests)
            or tuple(int(value) for value in input_requests if value is not None)
            != tuple(range(state_requests[0], state_requests[-1]))
            or not self.steps[-1].terminal_closure
        ):
            raise ValueError("controlled-I/O input requests/terminal closure differ")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("controlled-I/O operator topology must remain outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("controlled-I/O operator topology must remain prospective")


@dataclass(frozen=True, slots=True)
class StateCoordinateNormalization(CanonicalRecord):
    """One source-native coordinate mapped to one dimensionless deviation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/state-coordinate-normalization'

    mapping_id: str
    source_registration_id: str
    source_registration_ordinal: int
    source_label: str
    source_schema_sha256: str
    coordinate_id: str
    source_quantity_id: str
    source_native_unit: str
    source_native_frame: str
    normalization_scale: NamedDecimal
    normalized_frame: str

    def __post_init__(self) -> None:
        for name, value in (
            ("mapping_id", self.mapping_id),
            ("source_registration_id", self.source_registration_id),
            ("coordinate_id", self.coordinate_id),
            ("source_quantity_id", self.source_quantity_id),
            ("source_native_frame", self.source_native_frame),
            ("normalized_frame", self.normalized_frame),
        ):
            validate_stable_id(value, field_name=name)
        if self.source_registration_ordinal < 0:
            raise ValueError("source registration ordinal must be nonnegative")
        validate_nonempty(self.source_label, field_name="source_label")
        validate_sha256(self.source_schema_sha256, field_name="source_schema_sha256")
        if not self.source_native_unit.strip():
            raise ValueError("source native unit cannot be empty")
        if self.normalization_scale.unit != self.source_native_unit:
            raise ValueError("state normalization scale changes its source unit")
        validate_decimal(
            self.normalization_scale.value,
            field_name="normalization_scale",
            minimum=Decimal(0),
        )
        if self.normalization_scale.value == 0:
            raise ValueError("state normalization scale must be positive")


@dataclass(frozen=True, slots=True)
class NativeCoordinateCompatibility(CanonicalRecord):
    """One source-native coordinate retained without a unit/frame change."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/native-coordinate-compatibility'

    compatibility_id: str
    source_registration_id: str
    source_registration_ordinal: int
    source_label: str
    source_schema_sha256: str
    coordinate_id: str
    source_quantity_id: str
    source_native_unit: str
    source_native_frame: str
    controlled_native_unit: str
    controlled_native_frame: str
    transport_id: str

    def __post_init__(self) -> None:
        for field_name, field_value in (
            ("compatibility_id", self.compatibility_id),
            ("source_registration_id", self.source_registration_id),
            ("coordinate_id", self.coordinate_id),
            ("source_quantity_id", self.source_quantity_id),
            ("source_native_frame", self.source_native_frame),
            ("controlled_native_frame", self.controlled_native_frame),
            ("transport_id", self.transport_id),
        ):
            validate_stable_id(field_value, field_name=field_name)
        if self.source_registration_ordinal < 0:
            raise ValueError("source registration ordinal must be nonnegative")
        validate_nonempty(self.source_label, field_name="source_label")
        validate_sha256(self.source_schema_sha256, field_name="source_schema_sha256")
        if not self.source_native_unit.strip() or not self.controlled_native_unit.strip():
            raise ValueError("native coordinate units cannot be empty")
        if (
            self.source_native_unit != self.controlled_native_unit
            or self.source_native_frame != self.controlled_native_frame
        ):
            raise ValueError("native coordinate compatibility cannot change unit or frame")


@dataclass(frozen=True, slots=True)
class ControlledIOBasisNormalization(CanonicalRecord):
    """Exact native-to-normalized basis compatibility, frozen before outcomes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-basis-normalization'

    normalization_id: str
    state_basis: CoordinateBasis
    input_basis: CoordinateBasis
    receiver_basis: CoordinateBasis
    state_coordinates: tuple[StateCoordinateNormalization, ...]
    input_coordinates: tuple[NativeCoordinateCompatibility, ...]
    receiver_coordinates: tuple[NativeCoordinateCompatibility, ...]
    source_schema_sha256: str
    rule_id: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.normalization_id, field_name="normalization_id")
        validate_stable_id(self.rule_id, field_name="rule_id")
        validate_sha256(self.source_schema_sha256, field_name="source_schema_sha256")
        require_sorted_unique_ids(
            self.state_coordinates,
            attribute="mapping_id",
            field_name="state_coordinates",
        )
        if tuple(value.coordinate_id for value in self.state_coordinates) != (
            self.state_basis.coordinate_ids
        ):
            raise ValueError("state normalization does not cover the exact ordered basis")
        if tuple(value.source_registration_ordinal for value in self.state_coordinates) != tuple(
            range(len(self.state_coordinates))
        ):
            raise ValueError("state normalization does not preserve source registration order")
        for field_name, source_strings in (
            (
                "state source registration IDs",
                tuple(value.source_registration_id for value in self.state_coordinates),
            ),
            ("state source labels", tuple(value.source_label for value in self.state_coordinates)),
            (
                "state source quantities",
                tuple(value.source_quantity_id for value in self.state_coordinates),
            ),
        ):
            if len(set(source_strings)) != len(source_strings):
                raise ValueError(f"{field_name} must be unique")
        if len({value.source_schema_sha256 for value in self.state_coordinates}) != 1:
            raise ValueError("state normalization changes its authenticated source schema")
        if any(
            value.source_schema_sha256 != self.source_schema_sha256
            for value in self.state_coordinates
        ):
            raise ValueError("state normalization differs from its source schema fingerprint")
        if any(unit != "1" for unit in self.state_basis.native_units):
            raise ValueError("normalized controlled state basis must use unit 1")
        if tuple(value.normalized_frame for value in self.state_coordinates) != (
            self.state_basis.native_frames
        ):
            raise ValueError("state normalization frames differ from the controlled basis")
        for field_name, compatibilities, basis in (
            ("input_coordinates", self.input_coordinates, self.input_basis),
            ("receiver_coordinates", self.receiver_coordinates, self.receiver_basis),
        ):
            require_sorted_unique_ids(
                compatibilities,
                attribute="compatibility_id",
                field_name=field_name,
            )
            if tuple(value.coordinate_id for value in compatibilities) != basis.coordinate_ids:
                raise ValueError(f"{field_name} does not cover the exact ordered basis")
            if tuple(value.source_registration_ordinal for value in compatibilities) != tuple(
                range(len(compatibilities))
            ):
                raise ValueError(f"{field_name} does not preserve registration order")
            for source_field_name, source_values in (
                (
                    "registration IDs",
                    tuple(value.source_registration_id for value in compatibilities),
                ),
                (
                    "source labels",
                    tuple(value.source_label for value in compatibilities),
                ),
                (
                    "source quantities",
                    tuple(value.source_quantity_id for value in compatibilities),
                ),
            ):
                if len(set(source_values)) != len(source_values):
                    raise ValueError(f"{field_name} {source_field_name} must be unique")
            if len({value.source_schema_sha256 for value in compatibilities}) != 1:
                raise ValueError(f"{field_name} changes its authenticated source schema")
            if any(
                value.source_schema_sha256 != self.source_schema_sha256 for value in compatibilities
            ):
                raise ValueError(f"{field_name} differs from its source schema fingerprint")
            if (
                tuple(value.controlled_native_unit for value in compatibilities)
                != basis.native_units
            ):
                raise ValueError(f"{field_name} changes the controlled basis units")
            if (
                tuple(value.controlled_native_frame for value in compatibilities)
                != basis.native_frames
            ):
                raise ValueError(f"{field_name} changes the controlled basis frames")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("basis normalization must be frozen outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("basis normalization must remain prospective")


@dataclass(frozen=True, slots=True)
class ControlledIOOperatorEvidence(CanonicalRecord):
    """Authenticated bounded source operands and held-out validation summaries."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-operator-evidence'

    evidence_id: str
    prepared_denominator_id: str
    denominator_member_id: str
    candidate_version_id: str
    qualification_view_ids: tuple[str, ...]
    support_cell_ids: tuple[str, ...]
    normalization: ObjectIdentity
    operator_topology: ObjectIdentity
    state_basis: CoordinateBasis
    input_basis: CoordinateBasis
    receiver_basis: CoordinateBasis
    steps: tuple[ControlledIOStep, ...]
    reference_trajectory: ObjectIdentity
    reference_action_word: ObjectIdentity
    action_words: tuple[OccurrenceActionWord, ...]
    retained_history_id: str
    horizon_id: str
    held_out_physical_unit_ids: tuple[str, ...]
    residual_norm: NamedDecimal
    active_prediction_error: NamedDecimal
    wrong_sign_prediction_error: NamedDecimal
    zero_input_hold_error: NamedDecimal
    first_operator_sha256: str
    repeated_operator_sha256: str
    largest_operator_artifact_bytes: int
    materialized_member_bytes: int
    whole_stage_bytes: int
    complete_delivery: bool
    source_evaluator: ExecutableReference
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    disposition: ControlledIOProductDisposition
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("evidence_id", self.evidence_id),
            ("prepared_denominator_id", self.prepared_denominator_id),
            ("denominator_member_id", self.denominator_member_id),
            ("candidate_version_id", self.candidate_version_id),
            ("retained_history_id", self.retained_history_id),
            ("horizon_id", self.horizon_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, values in (
            ("qualification_view_ids", self.qualification_view_ids),
            ("support_cell_ids", self.support_cell_ids),
            ("held_out_physical_unit_ids", self.held_out_physical_unit_ids),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(
                values,
                field_name=name,
                allow_empty=name == "reason_codes",
            )
        require_sorted_unique_ids(
            self.action_words,
            attribute="word_id",
            field_name="action_words",
        )
        require_sorted_unique_ids(
            self.input_artifacts,
            attribute="artifact_id",
            field_name="input_artifacts",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        validate_sha256(self.first_operator_sha256, field_name="first_operator_sha256")
        validate_sha256(
            self.repeated_operator_sha256,
            field_name="repeated_operator_sha256",
        )
        for field_name, byte_count in (
            ("largest_operator_artifact_bytes", self.largest_operator_artifact_bytes),
            ("materialized_member_bytes", self.materialized_member_bytes),
            ("whole_stage_bytes", self.whole_stage_bytes),
        ):
            if byte_count < 0:
                raise ValueError(f"{field_name} must be nonnegative")
        expected_units = (
            "1",
            self.receiver_basis.native_units[0],
            self.receiver_basis.native_units[0],
            self.receiver_basis.native_units[0],
        )
        for field_name, metric, expected_unit in zip(
            (
                "residual_norm",
                "active_prediction_error",
                "wrong_sign_prediction_error",
                "zero_input_hold_error",
            ),
            (
                self.residual_norm,
                self.active_prediction_error,
                self.wrong_sign_prediction_error,
                self.zero_input_hold_error,
            ),
            expected_units,
            strict=True,
        ):
            if metric.unit != expected_unit:
                raise ValueError(f"{field_name} uses another frozen metric unit")
            validate_decimal(metric.value, field_name=field_name, minimum=Decimal(0))
        artifacts = {value.artifact_id for value in self.input_artifacts}
        linked = {value for link in self.evidence_links for value in link.artifact_ids}
        if not artifacts or artifacts != linked:
            raise ValueError("controlled operator evidence artifacts and links differ")
        if self.source_evaluator.payload.artifact_id not in artifacts:
            raise ValueError("controlled operator evidence omits its source evaluator")
        if self.reference_action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("controlled operator reference is not an ActionWord")
        if self.normalization.object_schema != ControlledIOBasisNormalization.SCHEMA:
            raise ValueError("controlled operator evidence binds another normalization schema")
        if self.operator_topology.object_schema != ControlledIOOperatorTopology.SCHEMA:
            raise ValueError("controlled operator evidence binds another topology schema")
        identities = {
            ObjectIdentity.from_record(value.word_id, value) for value in self.action_words
        }
        if self.reference_action_word not in identities:
            raise ValueError("controlled operator reference word is outside its chart")
        if any(
            value.denominator_id != self.prepared_denominator_id
            or value.retained_history_id != self.retained_history_id
            or value.horizon_id != self.horizon_id
            for value in self.action_words
        ):
            raise ValueError("controlled operator action chart changes denominator/history/horizon")
        if self.disposition is ControlledIOProductDisposition.SUPPORTED:
            if not self.steps or self.reason_codes:
                raise ValueError("supported operator evidence requires steps and no reasons")
        elif not self.reason_codes:
            raise ValueError("unavailable operator evidence requires a reason")
        if self.steps:
            if tuple(value.step_index for value in self.steps) != tuple(range(len(self.steps))):
                raise ValueError("operator evidence steps are not contiguous")
            for step in self.steps:
                if (
                    step.state_transition.row_coordinate_ids != self.state_basis.coordinate_ids
                    or step.realized_input_map.column_coordinate_ids
                    != self.input_basis.coordinate_ids
                    or step.receiver_map.row_coordinate_ids != self.receiver_basis.coordinate_ids
                ):
                    raise ValueError("operator evidence changes a declared basis")
        if self.outcome_access not in {
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATOR_REVEAL,
        }:
            raise ValueError("operator evidence has an unsupported outcome-access lane")
        if self.visibility_ceiling not in {
            VisibilityCeiling.DEVELOPMENT_ONLY,
            VisibilityCeiling.PROSPECTIVE,
        }:
            raise ValueError("operator evidence visibility is not bounded")
        if self.evidence_ceiling not in {
            EvidenceCeiling.NON_PROMOTABLE,
            EvidenceCeiling.LOCAL_LAW,
        }:
            raise ValueError("operator evidence cannot claim admission/controller use")


@dataclass(frozen=True, slots=True)
class ControlledIOMemberConstructionConfig(CanonicalRecord):
    """Frozen qualification and resource contract for one exact member."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-member-construction-config'

    config_id: str
    member_record_id: str
    expected_prepared_denominator_id: str
    expected_denominator_member_id: str
    expected_candidate_version_id: str
    expected_qualification_view_ids: tuple[str, ...]
    expected_support_cell_ids: tuple[str, ...]
    expected_action_words: tuple[ObjectIdentity, ...]
    expected_retained_history_id: str
    expected_horizon_id: str
    expected_normalization: ObjectIdentity
    expected_operator_topology: ObjectIdentity
    expected_step_ids: tuple[str, ...]
    expected_state_clock_id: str
    expected_input_clock_id: str
    expected_receiver_clock_id: str
    expected_held_out_physical_unit_ids: tuple[str, ...]
    required_step_count: int
    rank_tolerance: Decimal
    maximum_zero_input_hold_error: Decimal
    maximum_state_dimension: int
    maximum_input_dimension: int
    maximum_receiver_dimension: int
    maximum_steps: int
    maximum_operator_artifact_bytes: int
    maximum_materialized_member_bytes: int
    maximum_whole_stage_bytes: int
    qualification: ControlledIOQualificationConfig
    implementation_id: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.member_record_id, field_name="member_record_id")
        validate_stable_id(self.implementation_id, field_name="implementation_id")
        for field_name, stable_value in (
            ("expected_prepared_denominator_id", self.expected_prepared_denominator_id),
            ("expected_denominator_member_id", self.expected_denominator_member_id),
            ("expected_candidate_version_id", self.expected_candidate_version_id),
            ("expected_retained_history_id", self.expected_retained_history_id),
            ("expected_horizon_id", self.expected_horizon_id),
            ("expected_state_clock_id", self.expected_state_clock_id),
            ("expected_input_clock_id", self.expected_input_clock_id),
            ("expected_receiver_clock_id", self.expected_receiver_clock_id),
        ):
            validate_stable_id(stable_value, field_name=field_name)
        for field_name, values in (
            ("expected_qualification_view_ids", self.expected_qualification_view_ids),
            ("expected_support_cell_ids", self.expected_support_cell_ids),
            ("expected_step_ids", self.expected_step_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
            for stable_value in values:
                validate_stable_id(stable_value, field_name=field_name)
        require_sorted_unique_ids(
            self.expected_action_words,
            attribute="object_id",
            field_name="expected_action_words",
        )
        if not self.expected_action_words or any(
            value.object_schema != OccurrenceActionWord.SCHEMA for value in self.expected_action_words
        ):
            raise ValueError("controlled member config requires exact ActionWord identities")
        if self.expected_normalization.object_schema != ControlledIOBasisNormalization.SCHEMA:
            raise ValueError("controlled member config binds another normalization schema")
        if self.expected_operator_topology.object_schema != ControlledIOOperatorTopology.SCHEMA:
            raise ValueError("controlled member config binds another operator topology schema")
        if self.qualification.clock_contract != self.expected_operator_topology:
            raise ValueError("controlled member clock contract changes its operator topology")
        require_sorted_unique_strings(
            self.expected_held_out_physical_unit_ids,
            field_name="expected_held_out_physical_unit_ids",
            allow_empty=False,
        )
        if self.required_step_count < 2 or self.required_step_count > self.maximum_steps:
            raise ValueError("controlled member required step count is outside its ceiling")
        if len(self.expected_step_ids) != self.required_step_count:
            raise ValueError("controlled member expected step roster has another size")
        validate_decimal(self.rank_tolerance, field_name="rank_tolerance", minimum=Decimal(0))
        if self.rank_tolerance == 0:
            raise ValueError("controlled member rank tolerance must be positive")
        validate_decimal(
            self.maximum_zero_input_hold_error,
            field_name="maximum_zero_input_hold_error",
            minimum=Decimal(0),
        )
        for name, value, hard_maximum in (
            (
                "maximum_state_dimension",
                self.maximum_state_dimension,
                MAX_CONTROLLED_IO_STATE_DIMENSION,
            ),
            (
                "maximum_input_dimension",
                self.maximum_input_dimension,
                MAX_CONTROLLED_IO_INPUT_DIMENSION,
            ),
            (
                "maximum_receiver_dimension",
                self.maximum_receiver_dimension,
                MAX_CONTROLLED_IO_RECEIVER_DIMENSION,
            ),
            ("maximum_steps", self.maximum_steps, MAX_CONTROLLED_IO_STEPS),
            (
                "maximum_operator_artifact_bytes",
                self.maximum_operator_artifact_bytes,
                MAX_OPERATOR_ARTIFACT_BYTES,
            ),
            (
                "maximum_materialized_member_bytes",
                self.maximum_materialized_member_bytes,
                MAX_MATERIALIZED_MEMBER_BYTES,
            ),
            ("maximum_whole_stage_bytes", self.maximum_whole_stage_bytes, MAX_OPERATOR_STAGE_BYTES),
        ):
            if value <= 0 or value > hard_maximum:
                raise ValueError(f"{name} exceeds the generic implementation ceiling")
        if (
            self.qualification.state_dimension > self.maximum_state_dimension
            or self.qualification.input_dimension > self.maximum_input_dimension
            or self.qualification.receiver_dimension > self.maximum_receiver_dimension
        ):
            raise ValueError("controlled qualification dimensions exceed construction ceilings")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("final controlled member construction is evaluator-reveal only")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("final controlled member construction must remain prospective")


@dataclass(frozen=True, slots=True)
class ControlledIOMemberConstructionReceipt(CanonicalRecord):
    """Qualification receipt used by the existing member without an identity cycle."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-member-construction-receipt'

    receipt_id: str
    member_record_id: str
    config: ObjectIdentity
    evidence: ObjectIdentity
    normalization: ObjectIdentity
    implementation_id: str
    implementation_sha256: str
    diagnostics: ControlledIOMemberDiagnostics
    disposition: ControlledIOProductDisposition
    reason_codes: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("member_record_id", self.member_record_id),
            ("implementation_id", self.implementation_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.config.object_schema != ControlledIOMemberConstructionConfig.SCHEMA:
            raise ValueError("construction receipt binds another config schema")
        if self.evidence.object_schema != ControlledIOOperatorEvidence.SCHEMA:
            raise ValueError("construction receipt binds another evidence schema")
        if self.normalization.object_schema != ControlledIOBasisNormalization.SCHEMA:
            raise ValueError("construction receipt binds another normalization schema")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is ControlledIOProductDisposition.SUPPORTED:
            if self.reason_codes or self.diagnostics.method_reason_codes:
                raise ValueError("supported construction receipt cannot carry reasons")
        elif not self.reason_codes or self.reason_codes != self.diagnostics.method_reason_codes:
            raise ValueError("failed construction receipt and diagnostics reasons differ")
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("member construction receipt cannot itself claim admission")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("member construction receipt has the wrong outcome access")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("member construction receipt must remain prospective")


@dataclass(frozen=True, slots=True)
class ControlledIOMemberConstructionResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-member-construction-result'

    result_id: str
    member: ControlledIOMember
    receipt: ControlledIOMemberConstructionReceipt

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.member.member_record_id != self.receipt.member_record_id:
            raise ValueError("controlled construction result member/receipt IDs differ")
        receipt_identity = ObjectIdentity.from_record(self.receipt.receipt_id, self.receipt)
        if self.member.qualification_receipts != (receipt_identity,):
            raise ValueError("controlled member does not bind its exact construction receipt")
        if self.member.disposition is not self.receipt.disposition:
            raise ValueError("controlled construction result dispositions differ")


def _finite_horizon_map(steps: tuple[ControlledIOStep, ...]) -> npt.NDArray[np.float64]:
    """Build the same lower-triangular finite map used by ControlledIOEvaluator."""

    horizon = len(steps) - 1
    receiver_dimension = steps[0].receiver_map.shape[0]
    input_dimension = steps[0].realized_input_map.shape[1]
    state_dimension = steps[0].state_transition.shape[0]
    rows: list[npt.NDArray[np.float64]] = []
    for receiver_index in range(1, horizon + 1):
        columns: list[npt.NDArray[np.float64]] = []
        for input_index in range(horizon):
            if input_index >= receiver_index:
                block = np.zeros((receiver_dimension, input_dimension), dtype=np.float64)
            else:
                transition = np.eye(state_dimension, dtype=np.float64)
                for transition_index in range(input_index + 1, receiver_index):
                    transition = steps[transition_index].state_transition.as_array() @ transition
                block = (
                    steps[receiver_index].receiver_map.as_array()
                    @ transition
                    @ steps[input_index].realized_input_map.as_array()
                )
            columns.append(block)
        rows.append(np.hstack(columns))
    return np.vstack(rows)


def _controllability_matrix(
    steps: tuple[ControlledIOStep, ...],
) -> npt.NDArray[np.float64]:
    horizon = len(steps) - 1
    state_dimension = steps[0].state_transition.shape[0]
    columns: list[npt.NDArray[np.float64]] = []
    for input_index in range(horizon):
        transition = np.eye(state_dimension, dtype=np.float64)
        for transition_index in range(input_index + 1, horizon):
            transition = steps[transition_index].state_transition.as_array() @ transition
        columns.append(transition @ steps[input_index].realized_input_map.as_array())
    return np.hstack(columns)


def _observability_matrix(
    steps: tuple[ControlledIOStep, ...],
) -> npt.NDArray[np.float64]:
    state_dimension = steps[0].state_transition.shape[0]
    transition = np.eye(state_dimension, dtype=np.float64)
    rows: list[npt.NDArray[np.float64]] = []
    for receiver_index, step in enumerate(steps):
        if receiver_index:
            transition = steps[receiver_index - 1].state_transition.as_array() @ transition
        rows.append(step.receiver_map.as_array() @ transition)
    return np.vstack(rows)


def _rank_and_condition(
    matrix: npt.NDArray[np.float64],
    tolerance: Decimal,
) -> tuple[int, Decimal | None]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0:
        return 0, None
    threshold = float(tolerance) * float(singular[0])
    rank = int(np.sum(singular > threshold))
    if rank < min(matrix.shape) or singular[-1] <= 0:
        return rank, None
    return rank, decimal_from_float(float(singular[0] / singular[-1]))


def construct_controlled_io_member(
    *,
    evidence: ControlledIOOperatorEvidence,
    normalization: ControlledIOBasisNormalization,
    config: ControlledIOMemberConstructionConfig,
    implementation_sha256: str,
) -> ControlledIOMemberConstructionResult:
    "Validate source operands and construct one existing member exactly once."

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    qualification = config.qualification
    normalization_identity = ObjectIdentity.from_record(
        normalization.normalization_id,
        normalization,
    )
    action_word_identities = tuple(
        ObjectIdentity.from_record(value.word_id, value) for value in evidence.action_words
    )
    if (
        evidence.prepared_denominator_id != config.expected_prepared_denominator_id
        or evidence.denominator_member_id != config.expected_denominator_member_id
        or evidence.candidate_version_id != config.expected_candidate_version_id
        or evidence.qualification_view_ids != config.expected_qualification_view_ids
        or evidence.support_cell_ids != config.expected_support_cell_ids
        or action_word_identities != config.expected_action_words
        or evidence.retained_history_id != config.expected_retained_history_id
        or evidence.horizon_id != config.expected_horizon_id
        or normalization_identity != config.expected_normalization
        or evidence.normalization != config.expected_normalization
        or evidence.operator_topology != config.expected_operator_topology
        or tuple(value.step_id for value in evidence.steps) != config.expected_step_ids
    ):
        raise ValueError("controlled source evidence changes its predeclared member lineage")
    if (
        evidence.state_basis != normalization.state_basis
        or evidence.input_basis != normalization.input_basis
        or evidence.receiver_basis != normalization.receiver_basis
    ):
        raise ValueError("controlled evidence and normalization bases differ")
    if (
        qualification.state_dimension != evidence.state_basis.dimension
        or qualification.input_dimension != evidence.input_basis.dimension
        or qualification.receiver_dimension != evidence.receiver_basis.dimension
        or qualification.reference_trajectory != evidence.reference_trajectory
        or qualification.reference_action_word != evidence.reference_action_word
    ):
        raise ValueError("controlled qualification and source evidence identities differ")
    if evidence.held_out_physical_unit_ids != config.expected_held_out_physical_unit_ids:
        raise ValueError("controlled evidence uses another held-out physical-unit roster")
    if any(
        value.state_clock_id != config.expected_state_clock_id
        or value.input_clock_id != config.expected_input_clock_id
        or value.receiver_clock_id != config.expected_receiver_clock_id
        for value in evidence.steps
    ):
        raise ValueError("controlled source evidence changes its predeclared step clocks")

    reasons = set(evidence.reason_codes)
    disposition = evidence.disposition
    steps = evidence.steps
    maximum_prediction_error = max(
        evidence.active_prediction_error.value,
        evidence.wrong_sign_prediction_error.value,
    )
    spectral_radius = Decimal(0)
    condition_number: Decimal | None = Decimal(0)
    controllability_rank = 0
    observability_rank = 0

    if disposition is ControlledIOProductDisposition.SUPPORTED:
        if (
            evidence.state_basis.dimension > config.maximum_state_dimension
            or evidence.input_basis.dimension > config.maximum_input_dimension
            or evidence.receiver_basis.dimension > config.maximum_receiver_dimension
            or len(steps) != config.required_step_count
            or len(steps) > config.maximum_steps
            or evidence.largest_operator_artifact_bytes > config.maximum_operator_artifact_bytes
            or evidence.materialized_member_bytes > config.maximum_materialized_member_bytes
            or evidence.whole_stage_bytes > config.maximum_whole_stage_bytes
        ):
            reasons.add("CONTROLLED_IO_REPRESENTATION_BOUND_EXCEEDED")
        if evidence.first_operator_sha256 != evidence.repeated_operator_sha256:
            reasons.add("CONTROLLED_IO_OPERATOR_REPEATABILITY_FAILED")
        if not evidence.complete_delivery:
            reasons.add("CONTROLLED_IO_VALIDATION_DELIVERY_INCOMPLETE")
        if evidence.residual_norm.value > qualification.maximum_residual_norm:
            reasons.add("CONTROLLED_IO_RESIDUAL_NORM_EXCEEDED")
        if maximum_prediction_error > qualification.maximum_held_out_prediction_error:
            reasons.add("CONTROLLED_IO_HELD_OUT_PREDICTION_ERROR_EXCEEDED")
        if evidence.zero_input_hold_error.value > config.maximum_zero_input_hold_error:
            reasons.add("CONTROLLED_IO_ZERO_INPUT_HOLD_CHECK_FAILED")

        if steps:
            try:
                state_arrays = tuple(step.state_transition.as_array() for step in steps)
                if any(not np.all(np.isfinite(value)) for value in state_arrays):
                    raise ValueError("nonfinite controlled state operator")
                spectral_radius = max(
                    decimal_from_float(float(np.max(np.abs(np.linalg.eigvals(value)))))
                    for value in state_arrays
                )
                finite_map = _finite_horizon_map(steps)
                if not np.all(np.isfinite(finite_map)):
                    raise ValueError("nonfinite finite-horizon map")
                _controlled_rank, condition_number = _rank_and_condition(
                    finite_map,
                    config.rank_tolerance,
                )
                controllability_rank, _ = _rank_and_condition(
                    _controllability_matrix(steps),
                    config.rank_tolerance,
                )
                observability_rank, _ = _rank_and_condition(
                    _observability_matrix(steps),
                    config.rank_tolerance,
                )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                reasons.add("CONTROLLED_IO_NUMERICAL_EVALUATION_UNAVAILABLE")
                disposition = ControlledIOProductDisposition.UNEVALUABLE
                condition_number = None
            else:
                if spectral_radius > qualification.maximum_spectral_radius:
                    reasons.add("CONTROLLED_IO_SPECTRAL_RADIUS_EXCEEDED")
                if (
                    condition_number is None
                    or condition_number > qualification.maximum_condition_number
                ):
                    reasons.add("CONTROLLED_IO_CONDITION_NUMBER_EXCEEDED")
                if controllability_rank < qualification.minimum_controllability_rank:
                    reasons.add("CONTROLLED_IO_CONTROLLABILITY_RANK_INSUFFICIENT")
                if observability_rank < qualification.minimum_observability_rank:
                    reasons.add("CONTROLLED_IO_OBSERVABILITY_RANK_INSUFFICIENT")
        if reasons:
            if disposition is ControlledIOProductDisposition.SUPPORTED:
                disposition = ControlledIOProductDisposition.NOT_SUPPORTED

    reason_codes = tuple(sorted(reasons))
    diagnostics = ControlledIOMemberDiagnostics(
        diagnostics_id=f"diagnostics.{config.member_record_id}",
        residual_norm=NamedDecimal(
            value_id=f"residual.{config.member_record_id}",
            value=evidence.residual_norm.value,
            unit="1",
        ),
        held_out_prediction_error=NamedDecimal(
            value_id=f"heldout.{config.member_record_id}",
            value=maximum_prediction_error,
            unit=evidence.active_prediction_error.unit,
        ),
        maximum_spectral_radius=NamedDecimal(
            value_id=f"spectral-radius.{config.member_record_id}",
            value=spectral_radius,
            unit="1",
        ),
        controllability_rank=controllability_rank,
        observability_rank=observability_rank,
        maximum_condition_number=NamedDecimal(
            value_id=f"condition.{config.member_record_id}",
            value=condition_number if condition_number is not None else Decimal("1e999"),
            unit="1",
        ),
        evidence_link_ids=tuple(value.link_id for value in evidence.evidence_links),
        method_reason_codes=reason_codes,
    )
    receipt = ControlledIOMemberConstructionReceipt(
        receipt_id=f"receipt.{config.member_record_id}",
        member_record_id=config.member_record_id,
        config=ObjectIdentity.from_record(config.config_id, config),
        evidence=ObjectIdentity.from_record(evidence.evidence_id, evidence),
        normalization=ObjectIdentity.from_record(
            normalization.normalization_id,
            normalization,
        ),
        implementation_id=config.implementation_id,
        implementation_sha256=implementation_sha256,
        diagnostics=diagnostics,
        disposition=disposition,
        reason_codes=reason_codes,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    member = ControlledIOMember(
        member_record_id=config.member_record_id,
        prepared_denominator_id=evidence.prepared_denominator_id,
        denominator_member_id=evidence.denominator_member_id,
        candidate_version_id=evidence.candidate_version_id,
        qualification_view_ids=evidence.qualification_view_ids,
        support_cell_ids=evidence.support_cell_ids,
        state_basis=evidence.state_basis,
        input_basis=evidence.input_basis,
        receiver_basis=evidence.receiver_basis,
        steps=steps,
        qualification_config=qualification,
        reference_trajectory=evidence.reference_trajectory,
        reference_action_word=evidence.reference_action_word,
        action_words=evidence.action_words,
        retained_history_id=evidence.retained_history_id,
        horizon_id=evidence.horizon_id,
        diagnostics=diagnostics,
        qualification_receipts=(ObjectIdentity.from_record(receipt.receipt_id, receipt),),
        disposition=disposition,
        reason_codes=reason_codes,
    )
    return ControlledIOMemberConstructionResult(
        result_id=f"result.{config.member_record_id}",
        member=member,
        receipt=receipt,
    )


__all__ = [
    'ControlledIOBasisNormalization',
    'ControlledIOClockAxis',
    'ControlledIOMemberConstructionConfig',
    'ControlledIOMemberConstructionReceipt',
    'ControlledIOMemberConstructionResult',
    'ControlledIOOperatorEvidence',
    'ControlledIOOperatorStepTopology',
    'ControlledIOOperatorTopology',
    "MAX_CONTROLLED_IO_INPUT_DIMENSION",
    "MAX_CONTROLLED_IO_RECEIVER_DIMENSION",
    "MAX_CONTROLLED_IO_STATE_DIMENSION",
    "MAX_CONTROLLED_IO_STEPS",
    "MAX_MATERIALIZED_MEMBER_BYTES",
    "MAX_OPERATOR_ARTIFACT_BYTES",
    "MAX_OPERATOR_STAGE_BYTES",
    'NativeCoordinateCompatibility',
    'StateCoordinateNormalization',
    'construct_controlled_io_member',
]
