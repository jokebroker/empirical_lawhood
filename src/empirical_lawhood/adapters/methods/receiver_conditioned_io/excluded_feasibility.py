"""Outcome-bounded feasibility checks for excluded controlled-I/O probes.

This module deliberately stops before controlled-member construction.  It
checks whether six predeclared excluded representatives demonstrate that an
already authenticated source can expose a bounded, repeatable controlled-I/O
operator representation.  The resulting receipt is development-visible but
``NON_PROMOTABLE`` and can be used only by a preissue branch selector.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ExecutableReference, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.preissue_branch_selection import PreissueOperatorFeasibilityDisposition, PreissueOperatorFeasibility

from .contracts import (
    ControlledIOProductDisposition,
    ControlledIOStep,
    CoordinateBasis,
    GammaRule,
    decimal_from_float,
)
from .member_construction import MAX_CONTROLLED_IO_INPUT_DIMENSION, MAX_CONTROLLED_IO_RECEIVER_DIMENSION, MAX_CONTROLLED_IO_STATE_DIMENSION, MAX_CONTROLLED_IO_STEPS, MAX_MATERIALIZED_MEMBER_BYTES, MAX_OPERATOR_ARTIFACT_BYTES, MAX_OPERATOR_STAGE_BYTES, ControlledIOBasisNormalization, ControlledIOOperatorEvidence, ControlledIOOperatorTopology


EXPECTED_EXCLUDED_REPRESENTATIVE_COUNT = 6
EXPECTED_EXCLUDED_TIER_COUNT = 3
EXPECTED_EXCLUDED_MEMBER_COUNT = 2
EXPECTED_EXCLUDED_CAMPAIGN_EPISODE_COUNT = 96
EXPECTED_CONTROLLED_IO_STEP_COUNT = 7
ACTION_WORD_INPUT_PROJECTION_SCHEMA = 'empirical-lawhood/geometry/action-word-input-projection-spec'

EXCLUDED_FEASIBILITY_MAXIMUM_TANGENT_RESIDUAL = Decimal("0.00085")
EXCLUDED_FEASIBILITY_MAXIMUM_PREDICTION_ERROR = Decimal("0.00085")
EXCLUDED_FEASIBILITY_MAXIMUM_SPECTRAL_RADIUS = Decimal("1.0")
EXCLUDED_FEASIBILITY_MAXIMUM_CONDITION_NUMBER = Decimal("1000")
EXCLUDED_FEASIBILITY_RELATIVE_RANK_TOLERANCE = Decimal("1e-10")

CONTROLLED_IO_OPERATOR_API_UNAVAILABLE = "CONTROLLED_IO_OPERATOR_API_UNAVAILABLE"
CONTROLLED_IO_DERIVATIVE_SURFACE_UNAVAILABLE = "CONTROLLED_IO_DERIVATIVE_SURFACE_UNAVAILABLE"
CONTROLLED_IO_STATE_OR_OBSERVATION_REPRESENTATION_UNAVAILABLE = (
    "CONTROLLED_IO_STATE_OR_OBSERVATION_REPRESENTATION_UNAVAILABLE"
)
CONTROLLED_IO_REPRESENTATION_BOUND_EXCEEDED = "CONTROLLED_IO_REPRESENTATION_BOUND_EXCEEDED"
CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_NOT_SUPPORTED = (
    "CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_NOT_SUPPORTED"
)
CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_UNEVALUABLE = (
    "CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_UNEVALUABLE"
)

CLOSED_EXCLUDED_FEASIBILITY_OBSTRUCTION_PRECEDENCE: tuple[str, ...] = (
    CONTROLLED_IO_OPERATOR_API_UNAVAILABLE,
    CONTROLLED_IO_DERIVATIVE_SURFACE_UNAVAILABLE,
    CONTROLLED_IO_STATE_OR_OBSERVATION_REPRESENTATION_UNAVAILABLE,
    CONTROLLED_IO_REPRESENTATION_BOUND_EXCEEDED,
    CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_NOT_SUPPORTED,
    CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_UNEVALUABLE,
)


class ControlledIOExcludedSourceDisposition(StrEnum):
    """Upstream source state known before numerical feasibility is attempted."""

    AVAILABLE = "AVAILABLE"
    OPERATOR_API_UNAVAILABLE = CONTROLLED_IO_OPERATOR_API_UNAVAILABLE
    DERIVATIVE_SURFACE_UNAVAILABLE = CONTROLLED_IO_DERIVATIVE_SURFACE_UNAVAILABLE
    STATE_OR_OBSERVATION_REPRESENTATION_UNAVAILABLE = (
        CONTROLLED_IO_STATE_OR_OBSERVATION_REPRESENTATION_UNAVAILABLE
    )
    REPRESENTATION_BOUND_EXCEEDED = CONTROLLED_IO_REPRESENTATION_BOUND_EXCEEDED
    ORDINARY_FAILURE = "ORDINARY_FAILURE"
    UNEVALUABLE = "UNEVALUABLE"


class ControlledIOExcludedFeasibilityDisposition(StrEnum):
    """One feasibility result; closed G2 values are intentionally literal."""

    FEASIBLE = "FEASIBLE"
    OPERATOR_API_UNAVAILABLE = CONTROLLED_IO_OPERATOR_API_UNAVAILABLE
    DERIVATIVE_SURFACE_UNAVAILABLE = CONTROLLED_IO_DERIVATIVE_SURFACE_UNAVAILABLE
    STATE_OR_OBSERVATION_REPRESENTATION_UNAVAILABLE = (
        CONTROLLED_IO_STATE_OR_OBSERVATION_REPRESENTATION_UNAVAILABLE
    )
    REPRESENTATION_BOUND_EXCEEDED = CONTROLLED_IO_REPRESENTATION_BOUND_EXCEEDED
    NUMERICAL_FEASIBILITY_NOT_SUPPORTED = CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_NOT_SUPPORTED
    NUMERICAL_FEASIBILITY_UNEVALUABLE = CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_UNEVALUABLE
    ORDINARY_FAILURE = "CONTROLLED_IO_EXCLUDED_ORDINARY_FAILURE"
    UNEVALUABLE = "UNEVALUABLE"


class ControlledIOExcludedEvaluationStage(StrEnum):
    PREISSUE_EXCLUDED_QUALIFICATION = "PREISSUE_EXCLUDED_QUALIFICATION"
    POSTISSUE_FORBIDDEN = "POSTISSUE_FORBIDDEN"


class ControlledIOExcludedOperandTopology(StrEnum):
    """Replication topology for the three action-conditioned trajectories.

    The excluded probe is a matched comparison over one preparation unit.  Its
    HOLD, active, and wrong-sign trajectories are nested conditions and must
    never be counted as three independent replicates.  Numerical-member views
    of that preparation are nested too.
    """

    MATCHED_CONDITIONS_ONE_PREPARATION_UNIT = "MATCHED_CONDITIONS_ONE_PREPARATION_UNIT"


@dataclass(frozen=True, slots=True)
class ControlledIOExcludedRepresentativeSpec(CanonicalRecord):
    """Prospective identity of one exact tier-by-member feasibility probe."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-excluded-representative-spec'

    representative_id: str
    tier_id: str
    denominator_member_id: str
    expected_operator_evidence_id: str
    expected_step_ids: tuple[str, ...]
    normalization: ObjectIdentity
    operator_topology: ObjectIdentity
    representation_contract: ObjectIdentity
    source_evaluator: ObjectIdentity
    source_readiness: ObjectIdentity
    hold_episode_id: str
    active_episode_id: str
    wrong_sign_episode_id: str
    hold_physical_independent_unit_id: str
    active_physical_independent_unit_id: str
    wrong_sign_physical_independent_unit_id: str
    operand_topology: ControlledIOExcludedOperandTopology

    def __post_init__(self) -> None:
        for name, stable_value in (
            ("representative_id", self.representative_id),
            ("tier_id", self.tier_id),
            ("denominator_member_id", self.denominator_member_id),
            ("expected_operator_evidence_id", self.expected_operator_evidence_id),
            ("hold_episode_id", self.hold_episode_id),
            ("active_episode_id", self.active_episode_id),
            ("wrong_sign_episode_id", self.wrong_sign_episode_id),
            (
                "hold_physical_independent_unit_id",
                self.hold_physical_independent_unit_id,
            ),
            (
                "active_physical_independent_unit_id",
                self.active_physical_independent_unit_id,
            ),
            (
                "wrong_sign_physical_independent_unit_id",
                self.wrong_sign_physical_independent_unit_id,
            ),
        ):
            validate_stable_id(stable_value, field_name=name)
        if len(self.expected_step_ids) != EXPECTED_CONTROLLED_IO_STEP_COUNT:
            raise ValueError("excluded representative requires exactly seven steps")
        require_sorted_unique_strings(
            self.expected_step_ids,
            field_name="expected_step_ids",
            allow_empty=False,
        )
        episode_ids = (
            self.active_episode_id,
            self.hold_episode_id,
            self.wrong_sign_episode_id,
        )
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("excluded active, HOLD and wrong-sign episodes must differ")
        physical_unit_ids = {
            self.active_physical_independent_unit_id,
            self.hold_physical_independent_unit_id,
            self.wrong_sign_physical_independent_unit_id,
        }
        if (
            self.operand_topology
            is not ControlledIOExcludedOperandTopology.MATCHED_CONDITIONS_ONE_PREPARATION_UNIT
            or len(physical_unit_ids) != 1
        ):
            raise ValueError(
                "excluded trajectories must be matched conditions on one preparation unit"
            )
        if self.normalization.object_schema != ControlledIOBasisNormalization.SCHEMA:
            raise ValueError("excluded representative binds another normalization schema")
        if self.operator_topology.object_schema != ControlledIOOperatorTopology.SCHEMA:
            raise ValueError("excluded representative binds another operator topology schema")
        if self.source_evaluator.object_schema != ExecutableReference.SCHEMA:
            raise ValueError("excluded representative binds another evaluator schema")


@dataclass(frozen=True, slots=True)
class ControlledIOExcludedFeasibilitySpec(CanonicalRecord):
    """Frozen six-probe, preissue-only operator feasibility contract."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-excluded-feasibility-spec'

    spec_id: str
    excluded_campaign_id: str
    excluded_campaign_episode_ids: tuple[str, ...]
    protected_parent_ids: tuple[str, ...]
    representatives: tuple[ControlledIOExcludedRepresentativeSpec, ...]
    state_basis: CoordinateBasis
    input_basis: CoordinateBasis
    receiver_basis: CoordinateBasis
    phase_accumulator_coordinate_id: str
    state_clock_id: str
    input_clock_id: str
    receiver_clock_id: str
    expected_action_word_ids: tuple[str, ...]
    reference_hold_action_word_id: str
    input_projections: tuple[ObjectIdentity, ...]
    required_step_count: int
    maximum_state_dimension: int
    maximum_input_dimension: int
    maximum_receiver_dimension: int
    maximum_steps: int
    maximum_operator_artifact_bytes: int
    maximum_materialized_member_bytes: int
    maximum_whole_stage_bytes: int
    maximum_tangent_residual: Decimal
    maximum_active_prediction_error: Decimal
    maximum_wrong_sign_prediction_error: Decimal
    maximum_zero_input_hold_error: Decimal
    maximum_spectral_radius: Decimal
    maximum_condition_number: Decimal
    minimum_controllability_rank: int
    minimum_observability_rank: int
    relative_rank_tolerance: Decimal
    implementation_id: str
    implementation_sha256: str
    evaluation_stage: ControlledIOExcludedEvaluationStage
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    controlled_member_construction_permitted: bool
    postissue_rescue_permitted: bool

    def __post_init__(self) -> None:
        for name, stable_value in (
            ("spec_id", self.spec_id),
            ("excluded_campaign_id", self.excluded_campaign_id),
            ("phase_accumulator_coordinate_id", self.phase_accumulator_coordinate_id),
            ("state_clock_id", self.state_clock_id),
            ("input_clock_id", self.input_clock_id),
            ("receiver_clock_id", self.receiver_clock_id),
            ("reference_hold_action_word_id", self.reference_hold_action_word_id),
            ("implementation_id", self.implementation_id),
        ):
            validate_stable_id(stable_value, field_name=name)
        require_sorted_unique_strings(
            self.excluded_campaign_episode_ids,
            field_name="excluded_campaign_episode_ids",
            allow_empty=False,
        )
        if len(self.excluded_campaign_episode_ids) != (EXPECTED_EXCLUDED_CAMPAIGN_EPISODE_COUNT):
            raise ValueError("excluded feasibility spec must bind exactly 96 Q episodes")
        require_sorted_unique_strings(
            self.protected_parent_ids,
            field_name="protected_parent_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.representatives,
            attribute="representative_id",
            field_name="representatives",
        )
        if len(self.representatives) != EXPECTED_EXCLUDED_REPRESENTATIVE_COUNT:
            raise ValueError("excluded feasibility requires exactly six representatives")
        tiers = tuple(sorted({value.tier_id for value in self.representatives}))
        members = tuple(sorted({value.denominator_member_id for value in self.representatives}))
        if len(tiers) != EXPECTED_EXCLUDED_TIER_COUNT:
            raise ValueError("excluded feasibility requires exactly three tiers")
        if len(members) != EXPECTED_EXCLUDED_MEMBER_COUNT:
            raise ValueError("excluded feasibility requires exactly two numerical members")
        pairs = {(value.tier_id, value.denominator_member_id) for value in self.representatives}
        if pairs != {(tier, member) for tier in tiers for member in members}:
            raise ValueError("excluded representatives must be the complete 3 x 2 product")
        expected_evidence_ids = tuple(
            value.expected_operator_evidence_id for value in self.representatives
        )
        if len(set(expected_evidence_ids)) != len(expected_evidence_ids):
            raise ValueError("excluded representatives must bind distinct operator evidence")
        operand_episode_ids = {
            episode_id
            for representative in self.representatives
            for episode_id in (
                representative.hold_episode_id,
                representative.active_episode_id,
                representative.wrong_sign_episode_id,
            )
        }
        if len(operand_episode_ids) != 3 * EXPECTED_EXCLUDED_REPRESENTATIVE_COUNT:
            raise ValueError("excluded representative episode operands must be disjoint")
        if not operand_episode_ids.issubset(set(self.excluded_campaign_episode_ids)):
            raise ValueError("excluded representative operand is outside the Q campaign")
        physical_units_by_tier = {
            tier: {
                representative.hold_physical_independent_unit_id
                for representative in self.representatives
                if representative.tier_id == tier
            }
            for tier in tiers
        }
        if (
            any(len(values) != 1 for values in physical_units_by_tier.values())
            or len(set().union(*physical_units_by_tier.values())) != EXPECTED_EXCLUDED_TIER_COUNT
        ):
            raise ValueError(
                "excluded preparation units must be tier-distinct with nested member views"
            )
        if self.state_basis.dimension < 2:
            raise ValueError("excluded state basis requires physical state plus accumulator")
        if self.input_basis.dimension != 1 or self.receiver_basis.dimension != 1:
            raise ValueError("excluded feasibility requires scalar input and receiver bases")
        if self.input_basis.dimension > self.maximum_input_dimension:
            raise ValueError("frozen input basis exceeds the representation ceiling")
        if self.receiver_basis.dimension > self.maximum_receiver_dimension:
            raise ValueError("frozen receiver basis exceeds the representation ceiling")
        if self.phase_accumulator_coordinate_id != self.state_basis.coordinate_ids[-1]:
            raise ValueError("phase accumulator must be the final frozen state coordinate")
        require_sorted_unique_strings(
            self.expected_action_word_ids,
            field_name="expected_action_word_ids",
            allow_empty=False,
        )
        if self.reference_hold_action_word_id not in self.expected_action_word_ids:
            raise ValueError("excluded reference HOLD is absent from the action roster")
        require_sorted_unique_ids(
            self.input_projections,
            attribute="object_id",
            field_name="input_projections",
        )
        if len(self.input_projections) != 3 or any(
            value.object_schema != ACTION_WORD_INPUT_PROJECTION_SCHEMA
            for value in self.input_projections
        ):
            raise ValueError("excluded feasibility requires three exact input projections")
        if self.required_step_count != EXPECTED_CONTROLLED_IO_STEP_COUNT:
            raise ValueError("excluded feasibility requires the exact seven-step horizon")
        for name, bound_value, hard_maximum in (
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
            (
                "maximum_whole_stage_bytes",
                self.maximum_whole_stage_bytes,
                MAX_OPERATOR_STAGE_BYTES,
            ),
        ):
            if bound_value <= 0 or bound_value > hard_maximum:
                raise ValueError(f"{name} exceeds the generic implementation ceiling")
        if self.required_step_count > self.maximum_steps:
            raise ValueError("required step count exceeds its implementation ceiling")
        for name, threshold, plan_maximum in (
            (
                "maximum_tangent_residual",
                self.maximum_tangent_residual,
                EXCLUDED_FEASIBILITY_MAXIMUM_TANGENT_RESIDUAL,
            ),
            (
                "maximum_active_prediction_error",
                self.maximum_active_prediction_error,
                EXCLUDED_FEASIBILITY_MAXIMUM_PREDICTION_ERROR,
            ),
            (
                "maximum_wrong_sign_prediction_error",
                self.maximum_wrong_sign_prediction_error,
                EXCLUDED_FEASIBILITY_MAXIMUM_PREDICTION_ERROR,
            ),
            (
                "maximum_spectral_radius",
                self.maximum_spectral_radius,
                EXCLUDED_FEASIBILITY_MAXIMUM_SPECTRAL_RADIUS,
            ),
            (
                "maximum_condition_number",
                self.maximum_condition_number,
                EXCLUDED_FEASIBILITY_MAXIMUM_CONDITION_NUMBER,
            ),
            (
                "relative_rank_tolerance",
                self.relative_rank_tolerance,
                EXCLUDED_FEASIBILITY_RELATIVE_RANK_TOLERANCE,
            ),
        ):
            validate_decimal(threshold, field_name=name, minimum=Decimal(0))
            if threshold > plan_maximum:
                raise ValueError(f'{name} weakens the frozen tokamak control replication feasibility threshold')
        validate_decimal(
            self.maximum_zero_input_hold_error,
            field_name="maximum_zero_input_hold_error",
            minimum=Decimal(0),
        )
        if self.maximum_zero_input_hold_error != 0:
            raise ValueError("excluded native-HOLD self-check must require exact zero")
        if self.maximum_spectral_radius <= 0 or self.maximum_condition_number < 1:
            raise ValueError("excluded spectral/condition thresholds are invalid")
        if self.relative_rank_tolerance <= 0:
            raise ValueError("excluded relative rank tolerance must be positive")
        if self.minimum_controllability_rank < 1 or self.minimum_observability_rank < 1:
            raise ValueError("excluded feasibility requires positive control/observation rank")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.evaluation_stage is not (
            ControlledIOExcludedEvaluationStage.PREISSUE_EXCLUDED_QUALIFICATION
        ):
            raise ValueError("excluded feasibility spec cannot be authored postissue")
        if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("excluded feasibility is permanently nonpromotable")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("excluded feasibility spec must freeze before Q outcomes")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("excluded feasibility spec must remain prospective")
        if self.controlled_member_construction_permitted:
            raise ValueError("excluded feasibility cannot construct a controlled member")
        if self.postissue_rescue_permitted:
            raise ValueError("excluded feasibility cannot authorize postissue rescue")


@dataclass(frozen=True, slots=True)
class ControlledIOExcludedDeliveryIntegrity(CanonicalRecord):
    """Four-stage integrity and disqualifying delivery facts for one probe."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-excluded-delivery-integrity'

    delivery_id: str
    requested_complete: bool
    accepted_complete: bool
    applied_complete: bool
    realized_complete: bool
    missing_operand: bool
    clipped_operand: bool
    rejected_operand: bool
    foreign_operand: bool
    early_terminated_operand: bool
    evidence_link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.delivery_id, field_name="delivery_id")
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
            allow_empty=False,
        )

    @property
    def is_complete(self) -> bool:
        return (
            self.requested_complete
            and self.accepted_complete
            and self.applied_complete
            and self.realized_complete
            and not self.missing_operand
            and not self.clipped_operand
            and not self.rejected_operand
            and not self.foreign_operand
            and not self.early_terminated_operand
        )


@dataclass(frozen=True, slots=True)
class ControlledIOExcludedRepresentativeEvidence(CanonicalRecord):
    """Authenticated Q-only operands for one predeclared representative."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-excluded-representative-evidence'

    representative_id: str
    tier_id: str
    denominator_member_id: str
    source_readiness: ObjectIdentity
    operator_evidence: ControlledIOOperatorEvidence | None
    normalization: ControlledIOBasisNormalization | None
    representation_contract: ObjectIdentity | None
    input_projections: tuple[ObjectIdentity, ...]
    hold_episode_id: str
    active_episode_id: str
    wrong_sign_episode_id: str
    hold_physical_independent_unit_id: str
    active_physical_independent_unit_id: str
    wrong_sign_physical_independent_unit_id: str
    operand_topology: ControlledIOExcludedOperandTopology
    delivery: ControlledIOExcludedDeliveryIntegrity | None
    first_source_evaluation: ObjectIdentity | None
    repeated_source_evaluation: ObjectIdentity | None
    source_disposition: ControlledIOExcludedSourceDisposition
    source_reason_codes: tuple[str, ...]
    protected_parent_outcome_access: OutcomeAccess
    excluded_outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        for name, stable_value in (
            ("representative_id", self.representative_id),
            ("tier_id", self.tier_id),
            ("denominator_member_id", self.denominator_member_id),
            ("hold_episode_id", self.hold_episode_id),
            ("active_episode_id", self.active_episode_id),
            ("wrong_sign_episode_id", self.wrong_sign_episode_id),
            (
                "hold_physical_independent_unit_id",
                self.hold_physical_independent_unit_id,
            ),
            (
                "active_physical_independent_unit_id",
                self.active_physical_independent_unit_id,
            ),
            (
                "wrong_sign_physical_independent_unit_id",
                self.wrong_sign_physical_independent_unit_id,
            ),
        ):
            validate_stable_id(stable_value, field_name=name)
        require_sorted_unique_strings(self.source_reason_codes, field_name="source_reason_codes")
        if (
            self.operand_topology
            is not ControlledIOExcludedOperandTopology.MATCHED_CONDITIONS_ONE_PREPARATION_UNIT
            or len(
                {
                    self.hold_physical_independent_unit_id,
                    self.active_physical_independent_unit_id,
                    self.wrong_sign_physical_independent_unit_id,
                }
            )
            != 1
        ):
            raise ValueError(
                "excluded evidence must retain matched conditions on one preparation unit"
            )
        if self.source_disposition is ControlledIOExcludedSourceDisposition.AVAILABLE:
            if self.source_reason_codes:
                raise ValueError("available excluded source cannot carry source reasons")
            if (
                self.operator_evidence is None
                or self.normalization is None
                or self.representation_contract is None
                or self.delivery is None
                or self.first_source_evaluation is None
                or self.repeated_source_evaluation is None
            ):
                raise ValueError("available excluded source requires complete operands")
            if self.operator_evidence.disposition is not ControlledIOProductDisposition.SUPPORTED:
                raise ValueError("available excluded source requires bounded operator evidence")
        else:
            if not self.source_reason_codes:
                raise ValueError("unavailable excluded source requires a decisive reason")
            if (
                self.operator_evidence is not None
                and self.operator_evidence.disposition is ControlledIOProductDisposition.SUPPORTED
            ):
                raise ValueError("unavailable excluded source cannot carry supported evidence")
        if self.operator_evidence is not None:
            if self.operator_evidence.denominator_member_id != self.denominator_member_id:
                raise ValueError("excluded evidence changes its numerical member")
            if self.operator_evidence.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
                raise ValueError("excluded operator evidence must remain nonpromotable")
            if self.operator_evidence.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
                raise ValueError("excluded Q operator evidence must use the development lane")
            if self.operator_evidence.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY:
                raise ValueError("excluded Q operator evidence must remain development-only")
        if self.operator_evidence is not None and self.delivery is not None:
            if self.operator_evidence.complete_delivery != self.delivery.is_complete:
                raise ValueError("excluded delivery summary differs from its four-stage record")
        if self.first_source_evaluation is not None and self.repeated_source_evaluation is not None:
            if self.first_source_evaluation.object_id == self.repeated_source_evaluation.object_id:
                raise ValueError("excluded repeatability requires two distinct source invocations")
            if (
                self.first_source_evaluation.object_schema
                != self.repeated_source_evaluation.object_schema
            ):
                raise ValueError("excluded source invocation schemas differ")
        require_sorted_unique_ids(
            self.input_projections,
            attribute="object_id",
            field_name="input_projections",
        )
        if len(self.input_projections) != 3 or any(
            value.object_schema != ACTION_WORD_INPUT_PROJECTION_SCHEMA
            for value in self.input_projections
        ):
            raise ValueError("excluded evidence requires three exact input projections")
        if self.protected_parent_outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("excluded evidence cannot access protected parent outcomes")
        if self.excluded_outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("excluded Q evidence must declare development visibility")
        if self.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY:
            raise ValueError("excluded evidence cannot be promoted from development")
        if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("excluded representative evidence is permanently nonpromotable")


@dataclass(frozen=True, slots=True)
class ControlledIOExcludedRepresentativeResult(CanonicalRecord):
    """One retained calculation, including all decisive feasibility facts."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-excluded-representative-result'

    result_id: str
    representative_id: str
    tier_id: str
    denominator_member_id: str
    source_readiness: ObjectIdentity
    operator_evidence: ObjectIdentity | None
    normalization: ObjectIdentity | None
    representation_contract: ObjectIdentity | None
    delivery: ObjectIdentity | None
    maximum_tangent_residual: NamedDecimal | None
    active_prediction_error: NamedDecimal | None
    wrong_sign_prediction_error: NamedDecimal | None
    zero_input_hold_error: NamedDecimal | None
    maximum_spectral_radius: NamedDecimal | None
    finite_map_condition_number: NamedDecimal | None
    controllability_rank: int | None
    observability_rank: int | None
    operator_repeatable: bool
    representation_valid: bool
    resource_bounds_valid: bool
    delivery_complete: bool
    disposition: ControlledIOExcludedFeasibilityDisposition
    closed_obstruction_codes: tuple[str, ...]
    reason_codes: tuple[str, ...]
    evidence_link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, stable_value in (
            ("result_id", self.result_id),
            ("representative_id", self.representative_id),
            ("tier_id", self.tier_id),
            ("denominator_member_id", self.denominator_member_id),
        ):
            validate_stable_id(stable_value, field_name=name)
        for optional_metric in (
            self.maximum_tangent_residual,
            self.active_prediction_error,
            self.wrong_sign_prediction_error,
            self.zero_input_hold_error,
        ):
            if optional_metric is not None:
                validate_decimal(
                    optional_metric.value,
                    field_name=optional_metric.value_id,
                    minimum=Decimal(0),
                )
        for optional_metric in (
            self.maximum_spectral_radius,
            self.finite_map_condition_number,
        ):
            if optional_metric is not None:
                validate_decimal(
                    optional_metric.value,
                    field_name=optional_metric.value_id,
                    minimum=Decimal(0),
                )
        for name, rank in (
            ("controllability_rank", self.controllability_rank),
            ("observability_rank", self.observability_rank),
        ):
            if rank is not None and rank < 0:
                raise ValueError(f"{name} must be nonnegative when evaluable")
        _validate_closed_obstruction_order(self.closed_obstruction_codes)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
        )
        if self.disposition is ControlledIOExcludedFeasibilityDisposition.FEASIBLE:
            if self.closed_obstruction_codes or self.reason_codes:
                raise ValueError("feasible representative cannot carry reasons")
            if not all(
                (
                    self.operator_repeatable,
                    self.representation_valid,
                    self.resource_bounds_valid,
                    self.delivery_complete,
                )
            ):
                raise ValueError("feasible representative has an incomplete predicate")
            if any(
                value is None
                for value in (
                    self.operator_evidence,
                    self.normalization,
                    self.representation_contract,
                    self.delivery,
                    self.maximum_tangent_residual,
                    self.active_prediction_error,
                    self.wrong_sign_prediction_error,
                    self.zero_input_hold_error,
                    self.maximum_spectral_radius,
                    self.finite_map_condition_number,
                    self.controllability_rank,
                    self.observability_rank,
                )
            ):
                raise ValueError("feasible representative requires complete operands")
        elif not self.reason_codes:
            raise ValueError("non-feasible representative requires decisive reasons")


@dataclass(frozen=True, slots=True)
class ControlledIOExcludedFeasibilityReceipt(CanonicalRecord):
    """Complete six-record result usable only as preissue selection evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-excluded-feasibility-receipt'

    receipt_id: str
    spec: ObjectIdentity
    implementation_id: str
    implementation_sha256: str
    representative_results: tuple[ControlledIOExcludedRepresentativeResult, ...]
    disposition: ControlledIOExcludedFeasibilityDisposition
    closed_obstruction_codes: tuple[str, ...]
    reason_codes: tuple[str, ...]
    evidence_identities: tuple[ObjectIdentity, ...]
    evaluation_stage: ControlledIOExcludedEvaluationStage
    evidence_ceiling: EvidenceCeiling
    protected_parent_outcome_access: OutcomeAccess
    excluded_outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    controlled_member_construction_permitted: bool
    postissue_rescue_permitted: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.implementation_id, field_name="implementation_id")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.spec.object_schema != ControlledIOExcludedFeasibilitySpec.SCHEMA:
            raise ValueError("excluded receipt binds another specification schema")
        require_sorted_unique_ids(
            self.representative_results,
            attribute="result_id",
            field_name="representative_results",
        )
        if len(self.representative_results) != EXPECTED_EXCLUDED_REPRESENTATIVE_COUNT:
            raise ValueError("excluded receipt requires exactly six representative results")
        _validate_closed_obstruction_order(self.closed_obstruction_codes)
        expected_closed = _ordered_closed_codes(
            reason
            for result in self.representative_results
            for reason in result.closed_obstruction_codes
        )
        if self.closed_obstruction_codes != expected_closed:
            raise ValueError("excluded receipt omits a representative G2 obstruction")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected_reasons = tuple(
            sorted(
                {reason for result in self.representative_results for reason in result.reason_codes}
            )
        )
        if self.reason_codes != expected_reasons:
            raise ValueError("excluded receipt reason ledger is incomplete")
        require_sorted_unique_ids(
            self.evidence_identities,
            attribute="object_id",
            field_name="evidence_identities",
        )
        expected_evidence = tuple(
            sorted(
                (
                    identity
                    for result in self.representative_results
                    for identity in (result.source_readiness, result.operator_evidence)
                    if identity is not None
                ),
                key=lambda value: value.object_id,
            )
        )
        if self.evidence_identities != expected_evidence:
            raise ValueError("excluded receipt evidence identity ledger is incomplete")
        expected_disposition = _aggregate_disposition(
            self.representative_results,
            self.closed_obstruction_codes,
        )
        if self.disposition is not expected_disposition:
            raise ValueError("excluded receipt disposition violates fixed precedence")
        if self.disposition is ControlledIOExcludedFeasibilityDisposition.FEASIBLE:
            if self.closed_obstruction_codes or self.reason_codes:
                raise ValueError("FEASIBLE excluded receipt cannot carry obstruction reasons")
        elif not self.reason_codes:
            raise ValueError("obstructed excluded receipt requires complete reasons")
        if self.evaluation_stage is not (
            ControlledIOExcludedEvaluationStage.PREISSUE_EXCLUDED_QUALIFICATION
        ):
            raise ValueError("excluded receipt cannot represent postissue rescue")
        if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("excluded feasibility receipt is permanently nonpromotable")
        if self.protected_parent_outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("excluded receipt cannot access protected parent outcomes")
        if self.excluded_outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("excluded receipt must disclose its Q-development access")
        if self.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY:
            raise ValueError("excluded receipt must remain development-only")
        if self.controlled_member_construction_permitted:
            raise ValueError("excluded receipt cannot qualify a controlled member")
        if self.postissue_rescue_permitted:
            raise ValueError("excluded receipt cannot trigger postissue rescue")


def _validate_closed_obstruction_order(values: tuple[str, ...]) -> None:
    if values != _ordered_closed_codes(values):
        raise ValueError("closed G2 obstructions violate the frozen precedence")


def to_preissue_operator_feasibility(
    receipt: ControlledIOExcludedFeasibilityReceipt,
    *,
    assessment_id: str,
) -> PreissueOperatorFeasibility:
    """Project the nonpromotable receipt into the selector's metadata-only input."""

    feasible = receipt.disposition is ControlledIOExcludedFeasibilityDisposition.FEASIBLE
    if feasible:
        reason_codes: tuple[str, ...] = ()
    elif receipt.closed_obstruction_codes:
        reason_codes = tuple(sorted(receipt.closed_obstruction_codes))
    else:
        # Ordinary or unlisted source failures must remain unlisted to the
        # selector, which therefore stops instead of activating HFR.
        reason_codes = receipt.reason_codes
    return PreissueOperatorFeasibility(
        assessment_id=assessment_id,
        specification=receipt.spec,
        receipt=ObjectIdentity.from_record(receipt.receipt_id, receipt),
        evaluator_implementation=ObjectIdentity(
            object_id=receipt.implementation_id,
            object_schema=('empirical-lawhood/methods/controlled-input-output/excluded-feasibility-evaluator'),
            object_version="1.0.0",
            object_fingerprint=receipt.implementation_sha256,
        ),
        disposition=(
            PreissueOperatorFeasibilityDisposition.FEASIBLE
            if feasible
            else PreissueOperatorFeasibilityDisposition.OBSTRUCTED
        ),
        reason_codes=reason_codes,
        evidence_identities=receipt.evidence_identities,
        outcome_access=receipt.protected_parent_outcome_access,
        raw_outcome_values_present=False,
        measurement_through_law_qualification_outcome_used=False,
        final_member_qualification_used=False,
    )


def _ordered_closed_codes(values: Iterable[str]) -> tuple[str, ...]:
    observed = set(values)
    if not observed.issubset(set(CLOSED_EXCLUDED_FEASIBILITY_OBSTRUCTION_PRECEDENCE)):
        raise ValueError("unlisted reason cannot enter the closed G2 obstruction set")
    return tuple(code for code in CLOSED_EXCLUDED_FEASIBILITY_OBSTRUCTION_PRECEDENCE if code in observed)


def _finite_horizon_map(steps: tuple[ControlledIOStep, ...]) -> npt.NDArray[np.float64]:
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
        for transition_index in range(input_index + 1, horizon + 1):
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
            transition = step.state_transition.as_array() @ transition
        rows.append(step.receiver_map.as_array() @ transition)
    return np.vstack(rows)


def _relative_rank(matrix: npt.NDArray[np.float64], tolerance: Decimal) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return 0
    return int(np.sum(singular_values > float(tolerance) * float(singular_values[0])))


def _representation_reasons(
    spec: ControlledIOExcludedFeasibilitySpec,
    representative_spec: ControlledIOExcludedRepresentativeSpec,
    evidence: ControlledIOExcludedRepresentativeEvidence,
) -> tuple[str, ...]:
    operator = evidence.operator_evidence
    normalization = evidence.normalization
    representation_contract = evidence.representation_contract
    if operator is None or normalization is None or representation_contract is None:
        return ("CONTROLLED_IO_EXCLUDED_REQUIRED_REPRESENTATION_OPERAND_MISSING",)
    reasons: set[str] = set()
    if (
        operator.evidence_id != representative_spec.expected_operator_evidence_id
        or operator.state_basis != spec.state_basis
        or operator.input_basis != spec.input_basis
        or operator.receiver_basis != spec.receiver_basis
        or normalization.state_basis != spec.state_basis
        or normalization.input_basis != spec.input_basis
        or normalization.receiver_basis != spec.receiver_basis
        or ObjectIdentity.from_record(
            normalization.normalization_id,
            normalization,
        )
        != representative_spec.normalization
        or operator.operator_topology != representative_spec.operator_topology
        or representation_contract != representative_spec.representation_contract
        or evidence.input_projections != spec.input_projections
        or ObjectIdentity.from_record(
            operator.source_evaluator.reference_id,
            operator.source_evaluator,
        )
        != representative_spec.source_evaluator
    ):
        reasons.add("CONTROLLED_IO_EXCLUDED_BASIS_OR_MAPPING_MISMATCH")
    steps = operator.steps
    if (
        len(steps) != spec.required_step_count
        or tuple(value.step_id for value in steps) != representative_spec.expected_step_ids
        or tuple(value.step_index for value in steps) != tuple(range(spec.required_step_count))
    ):
        reasons.add("CONTROLLED_IO_EXCLUDED_STEP_ROSTER_MISMATCH")
    if steps and any(
        step.state_clock_id != spec.state_clock_id
        or step.input_clock_id != spec.input_clock_id
        or step.receiver_clock_id != spec.receiver_clock_id
        or step.gamma_rule is not GammaRule.ZERO
        for step in steps
    ):
        reasons.add("CONTROLLED_IO_EXCLUDED_CLOCK_OR_GAMMA_MISMATCH")
    if steps and any(
        np.any(step.state_affine_term.as_array() != 0.0)
        or np.any(step.receiver_affine_term.as_array() != 0.0)
        or np.any(step.fixed_gamma_contribution.as_array() != 0.0)
        for step in steps
    ):
        reasons.add("CONTROLLED_IO_EXCLUDED_AFFINE_TERM_NOT_ZERO")
    if steps:
        terminal = steps[-1]
        state_dimension = spec.state_basis.dimension
        if not np.array_equal(
            terminal.state_transition.as_array(),
            np.eye(state_dimension, dtype=np.float64),
        ) or np.any(terminal.realized_input_map.as_array() != 0.0):
            reasons.add("CONTROLLED_IO_EXCLUDED_TERMINAL_CLOSURE_MISMATCH")
    action_word_ids = tuple(value.word_id for value in operator.action_words)
    if (
        action_word_ids != spec.expected_action_word_ids
        or operator.reference_action_word.object_id != spec.reference_hold_action_word_id
    ):
        reasons.add("CONTROLLED_IO_EXCLUDED_ACTION_CHART_MISMATCH")
    expected_operand_ids = tuple(
        sorted(
            {
                evidence.active_physical_independent_unit_id,
                evidence.hold_physical_independent_unit_id,
                evidence.wrong_sign_physical_independent_unit_id,
            }
        )
    )
    if operator.held_out_physical_unit_ids != expected_operand_ids:
        reasons.add("CONTROLLED_IO_EXCLUDED_OPERAND_ROSTER_MISMATCH")
    return tuple(sorted(reasons))


def _resource_reasons(
    spec: ControlledIOExcludedFeasibilitySpec,
    evidence: ControlledIOExcludedRepresentativeEvidence,
) -> tuple[str, ...]:
    operator = evidence.operator_evidence
    if operator is None:
        return ("CONTROLLED_IO_EXCLUDED_RESOURCE_OPERAND_MISSING",)
    reasons: set[str] = set()
    if operator.state_basis.dimension > spec.maximum_state_dimension:
        reasons.add("CONTROLLED_IO_EXCLUDED_STATE_DIMENSION_BOUND_EXCEEDED")
    if operator.input_basis.dimension > spec.maximum_input_dimension:
        reasons.add("CONTROLLED_IO_EXCLUDED_INPUT_DIMENSION_BOUND_EXCEEDED")
    if operator.receiver_basis.dimension > spec.maximum_receiver_dimension:
        reasons.add("CONTROLLED_IO_EXCLUDED_RECEIVER_DIMENSION_BOUND_EXCEEDED")
    if len(operator.steps) > spec.maximum_steps:
        reasons.add("CONTROLLED_IO_EXCLUDED_STEP_BOUND_EXCEEDED")
    if operator.largest_operator_artifact_bytes > spec.maximum_operator_artifact_bytes:
        reasons.add("CONTROLLED_IO_EXCLUDED_OPERATOR_ARTIFACT_BOUND_EXCEEDED")
    if operator.materialized_member_bytes > spec.maximum_materialized_member_bytes:
        reasons.add("CONTROLLED_IO_EXCLUDED_MATERIALIZED_MEMBER_BOUND_EXCEEDED")
    if operator.whole_stage_bytes > spec.maximum_whole_stage_bytes:
        reasons.add("CONTROLLED_IO_EXCLUDED_OPERATOR_STAGE_BOUND_EXCEEDED")
    return tuple(sorted(reasons))


def _delivery_reasons(
    delivery: ControlledIOExcludedDeliveryIntegrity,
) -> tuple[str, ...]:
    reasons: set[str] = set()
    for complete, suffix in (
        (delivery.requested_complete, "REQUESTED"),
        (delivery.accepted_complete, "ACCEPTED"),
        (delivery.applied_complete, "APPLIED"),
        (delivery.realized_complete, "REALIZED"),
    ):
        if not complete:
            reasons.add(f"CONTROLLED_IO_EXCLUDED_{suffix}_DELIVERY_INCOMPLETE")
    for present, suffix in (
        (delivery.missing_operand, "MISSING_OPERAND"),
        (delivery.clipped_operand, "CLIPPED_OPERAND"),
        (delivery.rejected_operand, "REJECTED_OPERAND"),
        (delivery.foreign_operand, "FOREIGN_OPERAND"),
        (delivery.early_terminated_operand, "EARLY_TERMINATED_OPERAND"),
    ):
        if present:
            reasons.add(f"CONTROLLED_IO_EXCLUDED_{suffix}")
    return tuple(sorted(reasons))


def _source_closed_code(
    disposition: ControlledIOExcludedSourceDisposition,
) -> str | None:
    mapping = {
        ControlledIOExcludedSourceDisposition.OPERATOR_API_UNAVAILABLE: (
            CONTROLLED_IO_OPERATOR_API_UNAVAILABLE
        ),
        ControlledIOExcludedSourceDisposition.DERIVATIVE_SURFACE_UNAVAILABLE: (
            CONTROLLED_IO_DERIVATIVE_SURFACE_UNAVAILABLE
        ),
        ControlledIOExcludedSourceDisposition.STATE_OR_OBSERVATION_REPRESENTATION_UNAVAILABLE: (
            CONTROLLED_IO_STATE_OR_OBSERVATION_REPRESENTATION_UNAVAILABLE
        ),
        ControlledIOExcludedSourceDisposition.REPRESENTATION_BOUND_EXCEEDED: (
            CONTROLLED_IO_REPRESENTATION_BOUND_EXCEEDED
        ),
    }
    return mapping.get(disposition)


def _representative_disposition(
    *,
    closed_codes: tuple[str, ...],
    ordinary_reasons: tuple[str, ...],
    generally_unevaluable: bool,
) -> ControlledIOExcludedFeasibilityDisposition:
    if ordinary_reasons:
        return ControlledIOExcludedFeasibilityDisposition.ORDINARY_FAILURE
    if generally_unevaluable:
        return ControlledIOExcludedFeasibilityDisposition.UNEVALUABLE
    if closed_codes:
        return ControlledIOExcludedFeasibilityDisposition(closed_codes[0])
    return ControlledIOExcludedFeasibilityDisposition.FEASIBLE


def _aggregate_disposition(
    results: tuple[ControlledIOExcludedRepresentativeResult, ...],
    closed_codes: tuple[str, ...],
) -> ControlledIOExcludedFeasibilityDisposition:
    if all(
        value.disposition is ControlledIOExcludedFeasibilityDisposition.FEASIBLE
        for value in results
    ):
        return ControlledIOExcludedFeasibilityDisposition.FEASIBLE
    if any(
        value.disposition is ControlledIOExcludedFeasibilityDisposition.ORDINARY_FAILURE
        for value in results
    ):
        return ControlledIOExcludedFeasibilityDisposition.ORDINARY_FAILURE
    if any(
        value.disposition is ControlledIOExcludedFeasibilityDisposition.UNEVALUABLE
        for value in results
    ):
        return ControlledIOExcludedFeasibilityDisposition.UNEVALUABLE
    if not closed_codes:
        raise ValueError("non-feasible excluded receipt has no controlling disposition")
    return ControlledIOExcludedFeasibilityDisposition(closed_codes[0])


class ControlledIOExcludedFeasibilityEvaluator:
    """Pure evaluator for the six exact excluded representatives."""

    def evaluate(
        self,
        *,
        spec: ControlledIOExcludedFeasibilitySpec,
        representatives: tuple[ControlledIOExcludedRepresentativeEvidence, ...],
        implementation_sha256: str,
    ) -> ControlledIOExcludedFeasibilityReceipt:
        validate_sha256(implementation_sha256, field_name="implementation_sha256")
        if implementation_sha256 != spec.implementation_sha256:
            raise ValueError("excluded evaluator implementation differs from the frozen spec")
        require_sorted_unique_ids(
            representatives,
            attribute="representative_id",
            field_name="representatives",
        )
        if len(representatives) != EXPECTED_EXCLUDED_REPRESENTATIVE_COUNT:
            raise ValueError("excluded evaluator requires exactly six evidence records")
        evidence_by_id = {value.representative_id: value for value in representatives}
        if set(evidence_by_id) != {value.representative_id for value in spec.representatives}:
            raise ValueError("excluded evidence roster differs from the frozen specification")

        results = tuple(
            self._evaluate_representative(
                spec=spec,
                representative_spec=representative_spec,
                evidence=evidence_by_id[representative_spec.representative_id],
            )
            for representative_spec in spec.representatives
        )
        closed_codes = _ordered_closed_codes(
            reason for result in results for reason in result.closed_obstruction_codes
        )
        reason_codes = tuple(
            sorted({reason for result in results for reason in result.reason_codes})
        )
        evidence_identities = tuple(
            sorted(
                (
                    identity
                    for result in results
                    for identity in (result.source_readiness, result.operator_evidence)
                    if identity is not None
                ),
                key=lambda value: value.object_id,
            )
        )
        return ControlledIOExcludedFeasibilityReceipt(
            receipt_id=f"receipt.{spec.spec_id}",
            spec=ObjectIdentity.from_record(spec.spec_id, spec),
            implementation_id=spec.implementation_id,
            implementation_sha256=implementation_sha256,
            representative_results=results,
            disposition=_aggregate_disposition(results, closed_codes),
            closed_obstruction_codes=closed_codes,
            reason_codes=reason_codes,
            evidence_identities=evidence_identities,
            evaluation_stage=(
                ControlledIOExcludedEvaluationStage.PREISSUE_EXCLUDED_QUALIFICATION
            ),
            evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
            protected_parent_outcome_access=OutcomeAccess.OUTCOME_BLIND,
            excluded_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
            controlled_member_construction_permitted=False,
            postissue_rescue_permitted=False,
        )

    def _evaluate_representative(
        self,
        *,
        spec: ControlledIOExcludedFeasibilitySpec,
        representative_spec: ControlledIOExcludedRepresentativeSpec,
        evidence: ControlledIOExcludedRepresentativeEvidence,
    ) -> ControlledIOExcludedRepresentativeResult:
        if (
            evidence.tier_id != representative_spec.tier_id
            or evidence.denominator_member_id != representative_spec.denominator_member_id
            or evidence.source_readiness != representative_spec.source_readiness
            or evidence.hold_episode_id != representative_spec.hold_episode_id
            or evidence.active_episode_id != representative_spec.active_episode_id
            or evidence.wrong_sign_episode_id != representative_spec.wrong_sign_episode_id
            or evidence.hold_physical_independent_unit_id
            != representative_spec.hold_physical_independent_unit_id
            or evidence.active_physical_independent_unit_id
            != representative_spec.active_physical_independent_unit_id
            or evidence.wrong_sign_physical_independent_unit_id
            != representative_spec.wrong_sign_physical_independent_unit_id
            or evidence.operand_topology is not representative_spec.operand_topology
        ):
            raise ValueError("excluded representative identity differs from its frozen spec")

        operator = evidence.operator_evidence
        closed_reasons: set[str] = set()
        detail_reasons: set[str] = set(evidence.source_reason_codes)
        ordinary_reasons: set[str] = set()
        generally_unevaluable = False
        representation_valid = False
        resource_bounds_valid = False
        operator_repeatable = bool(
            operator is not None
            and operator.first_operator_sha256 == operator.repeated_operator_sha256
        )
        maximum_spectral_radius: NamedDecimal | None = None
        finite_map_condition_number: NamedDecimal | None = None
        controllability_rank: int | None = None
        observability_rank: int | None = None

        source_closed = _source_closed_code(evidence.source_disposition)
        if source_closed is not None:
            closed_reasons.add(source_closed)
        elif (
            evidence.source_disposition is ControlledIOExcludedSourceDisposition.ORDINARY_FAILURE
        ):
            ordinary_reasons.update(evidence.source_reason_codes)
        elif evidence.source_disposition is ControlledIOExcludedSourceDisposition.UNEVALUABLE:
            generally_unevaluable = True

        delivery_reasons = (
            _delivery_reasons(evidence.delivery) if evidence.delivery is not None else ()
        )
        if delivery_reasons:
            ordinary_reasons.update(delivery_reasons)
            detail_reasons.update(delivery_reasons)

        if evidence.source_disposition is ControlledIOExcludedSourceDisposition.AVAILABLE:
            if (
                operator is None
                or evidence.normalization is None
                or evidence.representation_contract is None
                or evidence.delivery is None
            ):
                raise ValueError("available excluded source lost its validated operands")
            representation_reasons = _representation_reasons(
                spec,
                representative_spec,
                evidence,
            )
            representation_valid = not representation_reasons
            if representation_reasons:
                closed_reasons.add(CONTROLLED_IO_STATE_OR_OBSERVATION_REPRESENTATION_UNAVAILABLE)
                detail_reasons.update(representation_reasons)
            resource_reasons = _resource_reasons(spec, evidence)
            resource_bounds_valid = not resource_reasons
            if resource_reasons:
                closed_reasons.add(CONTROLLED_IO_REPRESENTATION_BOUND_EXCEEDED)
                detail_reasons.update(resource_reasons)

            if representation_valid and resource_bounds_valid:
                numerical_reasons: set[str] = set()
                if not operator_repeatable:
                    numerical_reasons.add("CONTROLLED_IO_OPERATOR_REPEATABILITY_FAILED")
                if operator.residual_norm.value > spec.maximum_tangent_residual:
                    numerical_reasons.add("CONTROLLED_IO_EXCLUDED_TANGENT_RESIDUAL_EXCEEDED")
                if operator.active_prediction_error.value > spec.maximum_active_prediction_error:
                    numerical_reasons.add("CONTROLLED_IO_EXCLUDED_ACTIVE_PREDICTION_ERROR_EXCEEDED")
                if (
                    operator.wrong_sign_prediction_error.value
                    > spec.maximum_wrong_sign_prediction_error
                ):
                    numerical_reasons.add(
                        "CONTROLLED_IO_EXCLUDED_WRONG_SIGN_PREDICTION_ERROR_EXCEEDED"
                    )
                if operator.zero_input_hold_error.value > spec.maximum_zero_input_hold_error:
                    numerical_reasons.add("CONTROLLED_IO_EXCLUDED_ZERO_INPUT_HOLD_CHECK_FAILED")
                try:
                    steps = operator.steps
                    spectral = max(
                        float(np.max(np.abs(np.linalg.eigvals(step.state_transition.as_array()))))
                        for step in steps
                    )
                    finite_map = _finite_horizon_map(steps)
                    condition = float(np.linalg.cond(finite_map, p=2))
                    controllability_rank = _relative_rank(
                        _controllability_matrix(steps),
                        spec.relative_rank_tolerance,
                    )
                    observability_rank = _relative_rank(
                        _observability_matrix(steps),
                        spec.relative_rank_tolerance,
                    )
                    maximum_spectral_radius = NamedDecimal(
                        value_id=f"spectral-radius.{evidence.representative_id}",
                        value=decimal_from_float(spectral),
                        unit="1",
                    )
                    if np.isfinite(condition):
                        finite_map_condition_number = NamedDecimal(
                            value_id=f"condition-number.{evidence.representative_id}",
                            value=decimal_from_float(condition),
                            unit="1",
                        )
                    if maximum_spectral_radius.value > spec.maximum_spectral_radius:
                        numerical_reasons.add("CONTROLLED_IO_EXCLUDED_SPECTRAL_RADIUS_EXCEEDED")
                    if (
                        finite_map_condition_number is None
                        or finite_map_condition_number.value > spec.maximum_condition_number
                    ):
                        numerical_reasons.add("CONTROLLED_IO_EXCLUDED_CONDITION_NUMBER_EXCEEDED")
                    if controllability_rank < spec.minimum_controllability_rank:
                        numerical_reasons.add(
                            "CONTROLLED_IO_EXCLUDED_CONTROLLABILITY_RANK_INSUFFICIENT"
                        )
                    if observability_rank < spec.minimum_observability_rank:
                        numerical_reasons.add(
                            "CONTROLLED_IO_EXCLUDED_OBSERVABILITY_RANK_INSUFFICIENT"
                        )
                except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
                    closed_reasons.add(CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_UNEVALUABLE)
                    detail_reasons.add("CONTROLLED_IO_EXCLUDED_NUMERICAL_CALCULATION_FAILED")
                if numerical_reasons:
                    closed_reasons.add(CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_NOT_SUPPORTED)
                    detail_reasons.update(numerical_reasons)

        closed_codes = _ordered_closed_codes(closed_reasons)
        detail_reasons.update(closed_codes)
        reason_codes = tuple(sorted(detail_reasons))
        disposition = _representative_disposition(
            closed_codes=closed_codes,
            ordinary_reasons=tuple(sorted(ordinary_reasons)),
            generally_unevaluable=generally_unevaluable,
        )
        evidence_link_set = (
            set() if operator is None else {value.link_id for value in operator.evidence_links}
        )
        if evidence.delivery is not None:
            evidence_link_set.update(evidence.delivery.evidence_link_ids)
        evidence_links = tuple(sorted(evidence_link_set))
        return ControlledIOExcludedRepresentativeResult(
            result_id=f"result.{evidence.representative_id}",
            representative_id=evidence.representative_id,
            tier_id=evidence.tier_id,
            denominator_member_id=evidence.denominator_member_id,
            source_readiness=evidence.source_readiness,
            operator_evidence=(
                None
                if operator is None
                else ObjectIdentity.from_record(operator.evidence_id, operator)
            ),
            normalization=(
                None
                if evidence.normalization is None
                else ObjectIdentity.from_record(
                    evidence.normalization.normalization_id,
                    evidence.normalization,
                )
            ),
            representation_contract=evidence.representation_contract,
            delivery=(
                None
                if evidence.delivery is None
                else ObjectIdentity.from_record(
                    evidence.delivery.delivery_id,
                    evidence.delivery,
                )
            ),
            maximum_tangent_residual=(None if operator is None else operator.residual_norm),
            active_prediction_error=(
                None if operator is None else operator.active_prediction_error
            ),
            wrong_sign_prediction_error=(
                None if operator is None else operator.wrong_sign_prediction_error
            ),
            zero_input_hold_error=(None if operator is None else operator.zero_input_hold_error),
            maximum_spectral_radius=maximum_spectral_radius,
            finite_map_condition_number=finite_map_condition_number,
            controllability_rank=controllability_rank,
            observability_rank=observability_rank,
            operator_repeatable=operator_repeatable,
            representation_valid=representation_valid,
            resource_bounds_valid=resource_bounds_valid,
            delivery_complete=bool(evidence.delivery is not None and evidence.delivery.is_complete),
            disposition=disposition,
            closed_obstruction_codes=closed_codes,
            reason_codes=reason_codes,
            evidence_link_ids=evidence_links,
        )


__all__ = [
    "ACTION_WORD_INPUT_PROJECTION_SCHEMA",
    'CLOSED_EXCLUDED_FEASIBILITY_OBSTRUCTION_PRECEDENCE',
    "CONTROLLED_IO_DERIVATIVE_SURFACE_UNAVAILABLE",
    "CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_NOT_SUPPORTED",
    "CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_UNEVALUABLE",
    "CONTROLLED_IO_OPERATOR_API_UNAVAILABLE",
    "CONTROLLED_IO_REPRESENTATION_BOUND_EXCEEDED",
    "CONTROLLED_IO_STATE_OR_OBSERVATION_REPRESENTATION_UNAVAILABLE",
    'ControlledIOExcludedDeliveryIntegrity',
    'ControlledIOExcludedEvaluationStage',
    'ControlledIOExcludedFeasibilityDisposition',
    'ControlledIOExcludedFeasibilityEvaluator',
    'ControlledIOExcludedFeasibilityReceipt',
    'ControlledIOExcludedFeasibilitySpec',
    'ControlledIOExcludedOperandTopology',
    'ControlledIOExcludedRepresentativeEvidence',
    'ControlledIOExcludedRepresentativeResult',
    'ControlledIOExcludedRepresentativeSpec',
    'ControlledIOExcludedSourceDisposition',
    "EXPECTED_CONTROLLED_IO_STEP_COUNT",
    "EXPECTED_EXCLUDED_CAMPAIGN_EPISODE_COUNT",
    "EXPECTED_EXCLUDED_REPRESENTATIVE_COUNT",
    'EXCLUDED_FEASIBILITY_MAXIMUM_CONDITION_NUMBER',
    'EXCLUDED_FEASIBILITY_MAXIMUM_PREDICTION_ERROR',
    'EXCLUDED_FEASIBILITY_MAXIMUM_SPECTRAL_RADIUS',
    'EXCLUDED_FEASIBILITY_MAXIMUM_TANGENT_RESIDUAL',
    'EXCLUDED_FEASIBILITY_RELATIVE_RANK_TOLERANCE',
    'to_preissue_operator_feasibility',
]
