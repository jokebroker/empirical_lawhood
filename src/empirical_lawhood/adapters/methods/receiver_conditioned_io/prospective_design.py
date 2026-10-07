"""Prospective controlled-I/O topology and member-construction templates.

The excluded Q design is fully outcome-blind, while the existing final member
construction config correctly names an authenticated reference trajectory and
therefore cannot be serialized before that trajectory exists.  This module
keeps those lifecycle stages separate: the template freezes all scientific and
resource choices, and a post-reveal binder supplies only the predeclared
trajectory identity before the existing constructor is invoked.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.adapters.geometry.controlled_io_reachability import (
    ActionWordInputProjectionSpec,
)
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)

from .contracts import ControlledIOQualificationConfig, GammaRule
from .excluded_feasibility import ControlledIOExcludedFeasibilitySpec
from .member_construction import MAX_CONTROLLED_IO_INPUT_DIMENSION, MAX_CONTROLLED_IO_RECEIVER_DIMENSION, MAX_CONTROLLED_IO_STATE_DIMENSION, MAX_CONTROLLED_IO_STEPS, MAX_MATERIALIZED_MEMBER_BYTES, MAX_OPERATOR_ARTIFACT_BYTES, MAX_OPERATOR_STAGE_BYTES, ControlledIOBasisNormalization, ControlledIOOperatorStepTopology, ControlledIOOperatorTopology, ControlledIOMemberConstructionConfig


MAX_CONTROLLED_IO_DESIGN_STEPS = 32
MAX_CONTROLLED_IO_DESIGN_ACTIONS = 32
MAX_CONTROLLED_IO_DESIGN_MEMBERS = 32

# Public planning name retained for the exact step record now owned alongside
# the member constructor that must enforce it.


class ControlledIOProspectiveActionRole(StrEnum):
    ACTIVE = "ACTIVE"
    HOLD = "HOLD"
    WRONG_SIGN = "WRONG_SIGN"
    FUTURE = "FUTURE"


@dataclass(frozen=True, slots=True)
class ControlledIOProspectiveActionBinding(CanonicalRecord):
    """One exact action role and its optional controlled-input projection."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-prospective-action-binding'

    binding_id: str
    role: ControlledIOProspectiveActionRole
    action_word: ObjectIdentity
    input_projection: ObjectIdentity | None

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("controlled-I/O action binding requires an ActionWord")
        requires_projection = self.role is not ControlledIOProspectiveActionRole.FUTURE
        if requires_projection != (self.input_projection is not None):
            raise ValueError("controlled-I/O future is excluded and all other roles project")
        if (
            self.input_projection is not None
            and self.input_projection.object_schema != ActionWordInputProjectionSpec.SCHEMA
        ):
            raise ValueError("controlled-I/O action binding names another projection schema")


@dataclass(frozen=True, slots=True)
class ControlledIOMemberConstructionTemplate(CanonicalRecord):
    """Outcome-blind final-member design with a future trajectory identity slot."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-member-construction-template'

    template_id: str
    member_record_id: str
    prepared_denominator_id: str
    denominator_member_id: str
    candidate_version_id: str
    qualification_view_ids: tuple[str, ...]
    support_cell_ids: tuple[str, ...]
    action_words: tuple[ObjectIdentity, ...]
    retained_history_id: str
    horizon_id: str
    normalization: ObjectIdentity
    operator_topology: ControlledIOOperatorTopology
    expected_held_out_physical_unit_ids: tuple[str, ...]
    expected_reference_trajectory_id: str
    expected_reference_trajectory_schema: str
    reference_action_word: ObjectIdentity
    clock_contract: ObjectIdentity
    required_step_count: int
    state_dimension: int
    input_dimension: int
    receiver_dimension: int
    maximum_residual_norm: Decimal
    maximum_held_out_prediction_error: Decimal
    maximum_spectral_radius: Decimal
    maximum_condition_number: Decimal
    minimum_controllability_rank: int
    minimum_observability_rank: int
    gamma_rule: GammaRule
    stability_rule_id: str
    rank_tolerance: Decimal
    maximum_zero_input_hold_error: Decimal
    maximum_state_dimension: int
    maximum_input_dimension: int
    maximum_receiver_dimension: int
    maximum_steps: int
    maximum_operator_artifact_bytes: int
    maximum_materialized_member_bytes: int
    maximum_whole_stage_bytes: int
    binder_implementation_id: str
    binder_implementation_sha256: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for field_name, value in (
            ("template_id", self.template_id),
            ("member_record_id", self.member_record_id),
            ("prepared_denominator_id", self.prepared_denominator_id),
            ("denominator_member_id", self.denominator_member_id),
            ("candidate_version_id", self.candidate_version_id),
            ("retained_history_id", self.retained_history_id),
            ("horizon_id", self.horizon_id),
            ("expected_reference_trajectory_id", self.expected_reference_trajectory_id),
            ("stability_rule_id", self.stability_rule_id),
            ("binder_implementation_id", self.binder_implementation_id),
        ):
            validate_stable_id(value, field_name=field_name)
        validate_schema(self.expected_reference_trajectory_schema)
        validate_sha256(
            self.binder_implementation_sha256,
            field_name="binder_implementation_sha256",
        )
        require_sorted_unique_strings(
            self.expected_held_out_physical_unit_ids,
            field_name="expected_held_out_physical_unit_ids",
            allow_empty=False,
        )
        for unit_id in self.expected_held_out_physical_unit_ids:
            validate_stable_id(unit_id, field_name="expected_held_out_physical_unit_ids")
        for field_name, values in (
            ("qualification_view_ids", self.qualification_view_ids),
            ("support_cell_ids", self.support_cell_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
            for value in values:
                validate_stable_id(value, field_name=field_name)
        require_sorted_unique_ids(
            self.action_words,
            attribute="object_id",
            field_name="action_words",
        )
        if not self.action_words or any(
            value.object_schema != OccurrenceActionWord.SCHEMA for value in self.action_words
        ):
            raise ValueError("controlled member template requires exact ActionWord identities")
        if self.normalization.object_schema != ControlledIOBasisNormalization.SCHEMA:
            raise ValueError("controlled member template binds another normalization schema")
        topology_identity = ObjectIdentity.from_record(
            self.operator_topology.topology_id,
            self.operator_topology,
        )
        if self.clock_contract != topology_identity:
            raise ValueError("controlled member clock contract changes its operator topology")
        if self.reference_action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("controlled member template requires an ActionWord reference")
        if self.reference_action_word not in self.action_words:
            raise ValueError("controlled member reference ActionWord is outside its chart")
        if self.required_step_count != len(self.operator_topology.steps):
            raise ValueError("controlled member template changes its operator step topology")
        if self.required_step_count < 2 or self.required_step_count > self.maximum_steps:
            raise ValueError("controlled member template step count is outside its ceiling")
        for field_name, dimension_value, maximum in (
            ("state_dimension", self.state_dimension, self.maximum_state_dimension),
            ("input_dimension", self.input_dimension, self.maximum_input_dimension),
            ("receiver_dimension", self.receiver_dimension, self.maximum_receiver_dimension),
        ):
            if dimension_value < 1 or dimension_value > maximum:
                raise ValueError(f"{field_name} is outside its construction ceiling")
        for field_name, threshold, minimum in (
            ("maximum_residual_norm", self.maximum_residual_norm, Decimal(0)),
            (
                "maximum_held_out_prediction_error",
                self.maximum_held_out_prediction_error,
                Decimal(0),
            ),
            ("maximum_spectral_radius", self.maximum_spectral_radius, Decimal(0)),
            ("maximum_condition_number", self.maximum_condition_number, Decimal(1)),
            ("rank_tolerance", self.rank_tolerance, Decimal(0)),
            (
                "maximum_zero_input_hold_error",
                self.maximum_zero_input_hold_error,
                Decimal(0),
            ),
        ):
            validate_decimal(threshold, field_name=field_name, minimum=minimum)
        if self.maximum_spectral_radius == 0 or self.rank_tolerance == 0:
            raise ValueError("controlled member stability/rank thresholds must be positive")
        if self.minimum_controllability_rank < 1 or self.minimum_observability_rank < 1:
            raise ValueError("controlled member template requires positive ranks")
        for field_name, bound_value, hard_maximum in (
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
            if bound_value < 1 or bound_value > hard_maximum:
                raise ValueError(f"{field_name} exceeds the generic implementation ceiling")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("controlled member template must remain outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("controlled member template must remain prospective")


@dataclass(frozen=True, slots=True)
class ControlledIOMemberConstructionTemplateBindingReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-member-construction-template-binding-receipt'
    )

    binding_id: str
    template: ObjectIdentity
    reference_trajectory: ObjectIdentity
    construction_config: ObjectIdentity
    binder_implementation_id: str
    binder_implementation_sha256: str
    operator_values_read: bool
    grants_member_qualification: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(
            self.binder_implementation_id,
            field_name="binder_implementation_id",
        )
        validate_sha256(
            self.binder_implementation_sha256,
            field_name="binder_implementation_sha256",
        )
        if self.template.object_schema != ControlledIOMemberConstructionTemplate.SCHEMA:
            raise ValueError("controlled member binding names another template schema")
        if self.construction_config.object_schema != ControlledIOMemberConstructionConfig.SCHEMA:
            raise ValueError("controlled member binding names another config schema")
        if self.operator_values_read or self.grants_member_qualification:
            raise ValueError(
                "controlled member template binding cannot inspect or qualify a member"
            )
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("controlled member template binding is evaluator-reveal only")


def bind_controlled_io_member_construction_template(
    *,
    binding_id: str,
    template: ControlledIOMemberConstructionTemplate,
    reference_trajectory: ObjectIdentity,
    binder_implementation_id: str,
    binder_implementation_sha256: str,
) -> tuple[
    ControlledIOMemberConstructionConfig,
    ControlledIOMemberConstructionTemplateBindingReceipt,
]:
    """Author the existing final construction config from one exact trajectory identity."""

    if (
        binder_implementation_id != template.binder_implementation_id
        or binder_implementation_sha256 != template.binder_implementation_sha256
    ):
        raise ValueError("controlled member template binder implementation differs")
    if (
        reference_trajectory.object_id != template.expected_reference_trajectory_id
        or reference_trajectory.object_schema != template.expected_reference_trajectory_schema
    ):
        raise ValueError("controlled member reference trajectory differs from its template")
    qualification = ControlledIOQualificationConfig(
        config_id=f"qualification.{template.template_id}",
        state_dimension=template.state_dimension,
        input_dimension=template.input_dimension,
        receiver_dimension=template.receiver_dimension,
        maximum_residual_norm=template.maximum_residual_norm,
        maximum_held_out_prediction_error=template.maximum_held_out_prediction_error,
        maximum_spectral_radius=template.maximum_spectral_radius,
        maximum_condition_number=template.maximum_condition_number,
        minimum_controllability_rank=template.minimum_controllability_rank,
        minimum_observability_rank=template.minimum_observability_rank,
        gamma_rule=template.gamma_rule,
        reference_trajectory=reference_trajectory,
        reference_action_word=template.reference_action_word,
        clock_contract=template.clock_contract,
        stability_rule_id=template.stability_rule_id,
    )
    config = ControlledIOMemberConstructionConfig(
        config_id=f"config.{template.template_id}",
        member_record_id=template.member_record_id,
        expected_prepared_denominator_id=template.prepared_denominator_id,
        expected_denominator_member_id=template.denominator_member_id,
        expected_candidate_version_id=template.candidate_version_id,
        expected_qualification_view_ids=template.qualification_view_ids,
        expected_support_cell_ids=template.support_cell_ids,
        expected_action_words=template.action_words,
        expected_retained_history_id=template.retained_history_id,
        expected_horizon_id=template.horizon_id,
        expected_normalization=template.normalization,
        expected_operator_topology=ObjectIdentity.from_record(
            template.operator_topology.topology_id,
            template.operator_topology,
        ),
        expected_step_ids=tuple(value.step_id for value in template.operator_topology.steps),
        expected_state_clock_id=template.operator_topology.state_clock.clock_id,
        expected_input_clock_id=template.operator_topology.input_clock.clock_id,
        expected_receiver_clock_id=template.operator_topology.receiver_clock.clock_id,
        expected_held_out_physical_unit_ids=template.expected_held_out_physical_unit_ids,
        required_step_count=template.required_step_count,
        rank_tolerance=template.rank_tolerance,
        maximum_zero_input_hold_error=template.maximum_zero_input_hold_error,
        maximum_state_dimension=template.maximum_state_dimension,
        maximum_input_dimension=template.maximum_input_dimension,
        maximum_receiver_dimension=template.maximum_receiver_dimension,
        maximum_steps=template.maximum_steps,
        maximum_operator_artifact_bytes=template.maximum_operator_artifact_bytes,
        maximum_materialized_member_bytes=template.maximum_materialized_member_bytes,
        maximum_whole_stage_bytes=template.maximum_whole_stage_bytes,
        qualification=qualification,
        implementation_id=template.binder_implementation_id,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    receipt = ControlledIOMemberConstructionTemplateBindingReceipt(
        binding_id=binding_id,
        template=ObjectIdentity.from_record(template.template_id, template),
        reference_trajectory=reference_trajectory,
        construction_config=ObjectIdentity.from_record(config.config_id, config),
        binder_implementation_id=binder_implementation_id,
        binder_implementation_sha256=binder_implementation_sha256,
        operator_values_read=False,
        grants_member_qualification=False,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )
    return config, receipt


@dataclass(frozen=True, slots=True)
class ControlledIOProspectiveDesign(CanonicalRecord):
    """Closed prospective Q plus final-member topology; contains no operator result."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-prospective-design'

    design_id: str
    normalization: ControlledIOBasisNormalization
    action_words: tuple[OccurrenceActionWord, ...]
    action_bindings: tuple[ControlledIOProspectiveActionBinding, ...]
    input_projections: tuple[ActionWordInputProjectionSpec, ...]
    operator_topologies: tuple[ControlledIOOperatorTopology, ...]
    excluded_feasibility: ControlledIOExcludedFeasibilitySpec
    required_denominator_member_ids: tuple[str, ...]
    required_support_cell_ids: tuple[str, ...]
    final_member_templates: tuple[ControlledIOMemberConstructionTemplate, ...]
    final_receiver_output_coordinate_id: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.design_id, field_name="design_id")
        validate_stable_id(
            self.final_receiver_output_coordinate_id,
            field_name="final_receiver_output_coordinate_id",
        )
        require_sorted_unique_ids(
            self.action_words,
            attribute="word_id",
            field_name="action_words",
        )
        require_sorted_unique_ids(
            self.action_bindings,
            attribute="binding_id",
            field_name="action_bindings",
        )
        require_sorted_unique_ids(
            self.input_projections,
            attribute="projection_id",
            field_name="input_projections",
        )
        require_sorted_unique_ids(
            self.operator_topologies,
            attribute="topology_id",
            field_name="operator_topologies",
        )
        require_sorted_unique_ids(
            self.final_member_templates,
            attribute="template_id",
            field_name="final_member_templates",
        )
        for field_name, values in (
            ("required_denominator_member_ids", self.required_denominator_member_ids),
            ("required_support_cell_ids", self.required_support_cell_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
            for value in values:
                validate_stable_id(value, field_name=field_name)
        if not (1 <= len(self.action_words) <= MAX_CONTROLLED_IO_DESIGN_ACTIONS):
            raise ValueError("controlled-I/O design action roster has an invalid size")
        if not self.operator_topologies or any(
            not (2 <= len(value.steps) <= MAX_CONTROLLED_IO_DESIGN_STEPS)
            for value in self.operator_topologies
        ):
            raise ValueError("controlled-I/O design step roster has an invalid size")
        if not (1 <= len(self.final_member_templates) <= MAX_CONTROLLED_IO_DESIGN_MEMBERS):
            raise ValueError("controlled-I/O design member roster has an invalid size")
        roles = tuple(sorted((value.role for value in self.action_bindings), key=lambda x: x.value))
        expected_roles = tuple(sorted(ControlledIOProspectiveActionRole, key=lambda x: x.value))
        if roles != expected_roles:
            raise ValueError("controlled-I/O design requires active/HOLD/wrong/future roles")
        words = {
            ObjectIdentity.from_record(value.word_id, value): value for value in self.action_words
        }
        projections = {
            ObjectIdentity.from_record(value.projection_id, value): value
            for value in self.input_projections
        }
        for binding in self.action_bindings:
            if binding.action_word not in words:
                raise ValueError("controlled-I/O action binding changes its ActionWord")
            if binding.input_projection is not None:
                try:
                    projection = projections[binding.input_projection]
                except KeyError as error:
                    raise ValueError(
                        "controlled-I/O action binding changes its projection"
                    ) from error
                if projection.action_word != binding.action_word:
                    raise ValueError("controlled-I/O projection changes the bound ActionWord")
        if set(projections) != {
            value.input_projection
            for value in self.action_bindings
            if value.input_projection is not None
        }:
            raise ValueError("controlled-I/O design has an unbound or missing projection")
        excluded = self.excluded_feasibility
        step_count = len(self.operator_topologies[0].steps)
        if (
            excluded.state_basis != self.normalization.state_basis
            or excluded.input_basis != self.normalization.input_basis
            or excluded.receiver_basis != self.normalization.receiver_basis
            or excluded.required_step_count != step_count
            or excluded.expected_action_word_ids
            != tuple(value.word_id for value in self.action_words)
            or excluded.input_projections
            != tuple(
                ObjectIdentity.from_record(value.projection_id, value)
                for value in self.input_projections
            )
            or excluded.phase_accumulator_coordinate_id
            != self.normalization.state_basis.coordinate_ids[-1]
        ):
            raise ValueError("controlled-I/O Q spec differs from the prospective design")
        structural_topology = tuple(
            (
                value.step_index,
                value.source_state_request_index,
                value.source_input_request_index,
                value.state_coordinate,
                value.input_coordinate,
                value.receiver_coordinate,
                value.terminal_closure,
            )
            for value in self.operator_topologies[0].steps
        )
        reference_clocks = (
            self.operator_topologies[0].state_clock,
            self.operator_topologies[0].input_clock,
            self.operator_topologies[0].receiver_clock,
        )
        for topology in self.operator_topologies:
            if (
                topology.state_clock.clock_id != excluded.state_clock_id
                or topology.input_clock.clock_id != excluded.input_clock_id
                or topology.receiver_clock.clock_id != excluded.receiver_clock_id
                or (
                    topology.state_clock,
                    topology.input_clock,
                    topology.receiver_clock,
                )
                != reference_clocks
                or len(topology.steps) != step_count
                or tuple(
                    (
                        value.step_index,
                        value.source_state_request_index,
                        value.source_input_request_index,
                        value.state_coordinate,
                        value.input_coordinate,
                        value.receiver_coordinate,
                        value.terminal_closure,
                    )
                    for value in topology.steps
                )
                != structural_topology
            ):
                raise ValueError("controlled-I/O member topologies change the Q design")
        normalization_identity = ObjectIdentity.from_record(
            self.normalization.normalization_id,
            self.normalization,
        )
        if any(value.normalization != normalization_identity for value in excluded.representatives):
            raise ValueError("controlled-I/O Q representative changes the normalization")
        hold = next(
            value
            for value in self.action_bindings
            if value.role is ControlledIOProspectiveActionRole.HOLD
        )
        member_ids = tuple(value.member_record_id for value in self.final_member_templates)
        held_out_units = tuple(
            unit
            for value in self.final_member_templates
            for unit in value.expected_held_out_physical_unit_ids
        )
        if len(set(member_ids)) != len(member_ids) or len(set(held_out_units)) != len(
            held_out_units
        ):
            raise ValueError("controlled-I/O final members reuse an identity or physical unit")
        expected_pairs = {
            (member_id, support_cell_id)
            for member_id in self.required_denominator_member_ids
            for support_cell_id in self.required_support_cell_ids
        }
        actual_pairs = {
            (value.denominator_member_id, value.support_cell_ids[0])
            for value in self.final_member_templates
            if len(value.support_cell_ids) == 1
        }
        if len(actual_pairs) != len(self.final_member_templates) or actual_pairs != expected_pairs:
            raise ValueError(
                "controlled-I/O final members do not close the declared member/support product"
            )
        normalization_identity = ObjectIdentity.from_record(
            self.normalization.normalization_id,
            self.normalization,
        )
        topology_identities = {
            ObjectIdentity.from_record(value.topology_id, value)
            for value in self.operator_topologies
        }
        word_identities = tuple(
            ObjectIdentity.from_record(value.word_id, value) for value in self.action_words
        )
        for template in self.final_member_templates:
            if (
                template.reference_action_word != hold.action_word
                or template.normalization != normalization_identity
                or ObjectIdentity.from_record(
                    template.operator_topology.topology_id,
                    template.operator_topology,
                )
                not in topology_identities
                or template.action_words != word_identities
                or template.required_step_count != step_count
                or template.state_dimension != self.normalization.state_basis.dimension
                or template.input_dimension != self.normalization.input_basis.dimension
                or template.receiver_dimension != self.normalization.receiver_basis.dimension
            ):
                raise ValueError("controlled-I/O final member template changes the design")
        if {
            ObjectIdentity.from_record(
                value.operator_topology.topology_id,
                value.operator_topology,
            )
            for value in self.final_member_templates
        } != topology_identities:
            raise ValueError("controlled-I/O design has an unbound or missing member topology")
        expected_output = (
            f"receiver-step-{self.operator_topologies[0].steps[-1].step_index:04d}."
            f"{self.normalization.receiver_basis.coordinate_ids[0]}"
        )
        if self.final_receiver_output_coordinate_id != expected_output:
            raise ValueError("controlled-I/O final receiver row identity differs")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("controlled-I/O prospective design must remain outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("controlled-I/O prospective design must remain prospective")


__all__ = [
    'ControlledIOMemberConstructionTemplateBindingReceipt',
    'ControlledIOMemberConstructionTemplate',
    'ControlledIOProspectiveActionBinding',
    'ControlledIOProspectiveActionRole',
    'ControlledIOProspectiveDesign',
    'ControlledIOOperatorStepTopology',
    'bind_controlled_io_member_construction_template',
]
