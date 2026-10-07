"""Mechanical lowering and condition accounting for linked campaign profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.linked_campaign import CompleteUnitReserveGroup, LinkedCampaignOwnerBinding, LinkedCampaignOwnerRole, LinkedCampaignProfile, LinkedCampaignStageRole
from empirical_lawhood.planning.response_experiment import ResponseAcquisitionGroup, ResponseQualificationConfig, ResponseAcquisitionView
from empirical_lawhood.planning.observation_order import ObservationAcquisitionGroup, ObservationNestedView

from .artifacts import ArtifactProfile
from .capabilities import CapabilityConfigRef, CapabilityPermission, CapabilityRegistry
from .candidate_compiler import CandidateGraphEdge, CandidateGraphNode, CandidateScientificGraph, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, StudyCompilationReport
from .conditional_children import ConditionalChildResolution, ConditionalChildResolver
from .plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificInputRole,
    ScientificStage,
)


_STAGE_ORDER = (
    LinkedCampaignStageRole.SOURCE_MATERIALIZATION,
    LinkedCampaignStageRole.EVIDENCE_PROJECTION,
    LinkedCampaignStageRole.METHOD_IDENTIFICATION,
    LinkedCampaignStageRole.LAW_QUALIFICATION,
    LinkedCampaignStageRole.ATLAS_ASSEMBLY,
    LinkedCampaignStageRole.ADMISSION_EVIDENCE,
    LinkedCampaignStageRole.ADMISSION,
    LinkedCampaignStageRole.REACHABILITY,
    LinkedCampaignStageRole.PROGRAMME_AUTHORING,
)

_STAGE_KIND = {
    LinkedCampaignStageRole.SOURCE_MATERIALIZATION: ScientificStage.PREPARE,
    LinkedCampaignStageRole.EVIDENCE_PROJECTION: ScientificStage.TRANSFORM,
    LinkedCampaignStageRole.METHOD_IDENTIFICATION: ScientificStage.DEVELOP,
    LinkedCampaignStageRole.LAW_QUALIFICATION: ScientificStage.QUALIFY,
    LinkedCampaignStageRole.ATLAS_ASSEMBLY: ScientificStage.SYNTHESIZE,
    LinkedCampaignStageRole.ADMISSION_EVIDENCE: ScientificStage.ADMISSION,
    LinkedCampaignStageRole.ADMISSION: ScientificStage.ADMISSION,
    LinkedCampaignStageRole.REACHABILITY: ScientificStage.ADMISSION,
    LinkedCampaignStageRole.PROGRAMME_AUTHORING: ScientificStage.CONTROLLER,
}

_OWNER_ORDER = tuple(LinkedCampaignOwnerRole)


def _stable_role(role: StrEnum) -> str:
    return role.value.lower().replace("_", "-")


@dataclass(frozen=True, slots=True)
class LinkedCampaignStageEnvelope(CanonicalRecord):
    """Typed stage union retaining a product, obstruction or condition-false stop."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/linked-campaign-stage-envelope'

    envelope_id: str
    role: LinkedCampaignStageRole
    disposition: LinkedCampaignDisposition
    scientific_product: ObjectIdentity | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.envelope_id, field_name="envelope_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is LinkedCampaignDisposition.SUPPORTED:
            if self.scientific_product is None or self.reason_codes:
                raise ValueError("supported linked stage requires a product and no reasons")
        elif self.disposition in {
            LinkedCampaignDisposition.CONDITION_FALSE,
            LinkedCampaignDisposition.AUTHORITY_REQUIRED,
        }:
            if self.scientific_product is not None or not self.reason_codes:
                raise ValueError("nonattempt linked stage requires reasons and no product")
        elif self.scientific_product is None or not self.reason_codes:
            raise ValueError(
                "negative/unevaluable linked stage retains its exact product and reason"
            )


class LinkedCampaignDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    SCIENTIFIC_NEGATIVE = "SCIENTIFIC_NEGATIVE"
    UNEVALUABLE = "UNEVALUABLE"
    CONDITION_FALSE = "CONDITION_FALSE"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"


class LinkedCampaignParentDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    ZERO_LAW = "ZERO_LAW"
    PARTIAL_UNEVALUABLE = "PARTIAL_UNEVALUABLE"


class LinkedCampaignControllerUseQualification(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_REACHED = "NOT_REACHED"


@dataclass(frozen=True, slots=True)
class LinkedCampaignGateState(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/linked-campaign-gate-state'

    state_id: str
    parent_disposition: LinkedCampaignParentDisposition
    admission_corpus_complete: bool
    admission_reachable: bool
    measured_hold_supported: bool
    controller_use_qualification: LinkedCampaignControllerUseQualification
    execution_authorized: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.state_id, field_name="state_id")
        parent_supported = self.parent_disposition is LinkedCampaignParentDisposition.SUPPORTED
        if not parent_supported and (
            self.admission_corpus_complete
            or self.admission_reachable
            or self.measured_hold_supported
            or self.controller_use_qualification is not LinkedCampaignControllerUseQualification.NOT_REACHED
        ):
            raise ValueError("condition-false descendants cannot claim downstream evidence")
        if self.admission_reachable and not self.admission_corpus_complete:
            raise ValueError("reachability cannot precede a complete admission corpus")
        if self.measured_hold_supported and not self.admission_reachable:
            raise ValueError("measured HOLD cannot bypass admission/reachability")
        if self.controller_use_qualification is not LinkedCampaignControllerUseQualification.NOT_REACHED and not (
            self.admission_reachable and self.measured_hold_supported
        ):
            raise ValueError("Controller-use qualification cannot bypass controller prerequisites")


@dataclass(frozen=True, slots=True)
class LinkedCampaignRoleDisposition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/linked-campaign-role-disposition'

    disposition_id: str
    role: LinkedCampaignOwnerRole
    disposition: LinkedCampaignDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.disposition_id, field_name="disposition_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is LinkedCampaignDisposition.SUPPORTED:
            if self.reason_codes:
                raise ValueError("supported linked role cannot carry reasons")
        elif not self.reason_codes:
            raise ValueError("non-supported linked role requires reasons")


@dataclass(frozen=True, slots=True)
class LinkedCampaignRouteDisposition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/linked-campaign-route-disposition'

    route_id: str
    profile: ObjectIdentity
    roles: tuple[LinkedCampaignRoleDisposition, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.route_id, field_name="route_id")
        if self.profile.object_schema != LinkedCampaignProfile.SCHEMA:
            raise ValueError("linked route disposition requires an exact profile")
        require_sorted_unique_ids(
            self.roles,
            attribute="disposition_id",
            field_name="roles",
        )
        if {value.role for value in self.roles} != set(_OWNER_ORDER) or len(self.roles) != len(
            _OWNER_ORDER
        ):
            raise ValueError("linked route must retain every current-owner role")


@dataclass(frozen=True, slots=True)
class ReserveSubstitution(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/reserve-substitution'

    substitution_id: str
    unavailable_primary_unit_id: str
    reserve_physical_unit_id: str
    reserve_rank: int

    def __post_init__(self) -> None:
        for name, value in (
            ("substitution_id", self.substitution_id),
            ("unavailable_primary_unit_id", self.unavailable_primary_unit_id),
            ("reserve_physical_unit_id", self.reserve_physical_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.reserve_rank < 0:
            raise ValueError("reserve substitution rank must be nonnegative")


@dataclass(frozen=True, slots=True)
class ReserveSubstitutionReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/reserve-substitution-receipt'

    receipt_id: str
    reserve_group: ObjectIdentity
    substitutions: tuple[ReserveSubstitution, ...]
    final_physical_unit_ids: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.reserve_group.object_schema != CompleteUnitReserveGroup.SCHEMA:
            raise ValueError("reserve receipt requires an exact complete-unit group")
        require_sorted_unique_ids(
            self.substitutions,
            attribute="substitution_id",
            field_name="substitutions",
        )
        require_sorted_unique_strings(
            self.final_physical_unit_ids,
            field_name="final_physical_unit_ids",
            allow_empty=False,
        )
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("reserve substitution must remain outcome-blind")


def build_linked_campaign_protocol(
    profile: LinkedCampaignProfile,
    registry: CapabilityRegistry,
) -> ProtocolTemplate:
    """Mechanically lower the fixed ordinary topology to the existing template."""

    by_role = {value.role: value for value in profile.capabilities}
    steps: list[ProtocolStepTemplate] = []
    previous_step_id: str | None = None
    for role in _STAGE_ORDER:
        binding = by_role[role]
        manifest = registry.resolve(
            binding.selection.capability_key,
            binding.selection.capability_version,
        )
        if manifest.implementation_sha256 != binding.selection.implementation_sha256:
            raise ValueError("linked campaign capability implementation drifted")
        if LinkedCampaignStageEnvelope.SCHEMA not in manifest.output_schema_ids:
            raise ValueError("linked campaign capability lacks the stage-envelope output")
        config = CapabilityConfigRef(
            config_id=binding.config_id,
            config_schema=binding.config_schema,
            config_schema_sha256=binding.config_schema_sha256,
            content_sha256=binding.config_content_sha256,
            artifact_id=binding.config_artifact_id,
        )
        step_id = f"linked-{_stable_role(role)}"
        permissions = tuple(
            value
            for value in (
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            )
            if value in manifest.permissions
        )
        steps.append(
            ProtocolStepTemplate(
                step_id=step_id,
                stage=_STAGE_KIND[role],
                capability_key=manifest.capability_key,
                capability_version=manifest.capability_version,
                config=config,
                dependency_step_ids=() if previous_step_id is None else (previous_step_id,),
                outputs=(
                    OutputTemplate(
                        output_id="stage-envelope",
                        payload_schema=LinkedCampaignStageEnvelope.SCHEMA,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type="application/json",
                        filename_suffix=".canonical.json",
                    ),
                ),
                required_permissions=permissions,
                requested_outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                resource_budget=binding.resource_budget,
                resource_lock_ids=binding.resource_lock_ids,
                barrier=BarrierKind.NONE,
                maximum_attempts=binding.maximum_attempts,
                obligation_ids=(f"linked-obligation.{_stable_role(role)}",),
            )
        )
        previous_step_id = step_id
    return ProtocolTemplate(
        template_id=f"protocol.{profile.profile_id}",
        template_version=profile.profile_version,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=True,
        nonactuating=True,
    )


def expand_linked_study_acquisition_views(
    *,
    template: ProtocolTemplate,
    identification_config: ResponseQualificationConfig,
    source_task_prefix: str,
    projection_task_prefix: str,
    source_outputs: tuple[OutputTemplate, ...] | None = None,
    projection_outputs: tuple[OutputTemplate, ...] | None = None,
    source_step_template: ProtocolStepTemplate | None = None,
    projection_step_template: ProtocolStepTemplate | None = None,
    source_config_by_group: dict[str, CapabilityConfigRef] | None = None,
    projection_config_by_view: dict[str, CapabilityConfigRef] | None = None,
    acquisition_aggregate_step_template: ProtocolStepTemplate | None = None,
    acquisition_aggregate_task_id: str | None = None,
) -> ProtocolTemplate:
    """Expand one source task per acquisition and one pure task per view.

    Task prefixes and step contracts are adapter-owned code inputs. The shared
    expansion knows only acquisition/view lineage and never branches on medium.

    ``source_outputs``/``projection_outputs`` retain the original generic-step
    composition for frozen callers. New production callers provide complete
    step templates so that capability, config, permissions, native schemas,
    locks and budgets remain adapter-owned instead of being copied from the
    generic linked placeholders.
    """

    return expand_acquisition_views(
        template=template,
        carrier_fingerprint=identification_config.fingerprint(),
        acquisition_groups=identification_config.acquisition_groups,
        nested_views=identification_config.nested_views,
        source_task_prefix=source_task_prefix,
        projection_task_prefix=projection_task_prefix,
        source_outputs=source_outputs,
        projection_outputs=projection_outputs,
        source_step_template=source_step_template,
        projection_step_template=projection_step_template,
        source_config_by_group=source_config_by_group,
        projection_config_by_view=projection_config_by_view,
        acquisition_aggregate_step_template=acquisition_aggregate_step_template,
        acquisition_aggregate_task_id=acquisition_aggregate_task_id,
    )


def expand_acquisition_views(
    *,
    template: ProtocolTemplate,
    carrier_fingerprint: str,
    acquisition_groups: tuple[ResponseAcquisitionGroup | ObservationAcquisitionGroup, ...],
    nested_views: tuple[ResponseAcquisitionView | ObservationNestedView, ...],
    source_task_prefix: str,
    projection_task_prefix: str,
    source_outputs: tuple[OutputTemplate, ...] | None = None,
    projection_outputs: tuple[OutputTemplate, ...] | None = None,
    source_step_template: ProtocolStepTemplate | None = None,
    projection_step_template: ProtocolStepTemplate | None = None,
    source_config_by_group: dict[str, CapabilityConfigRef] | None = None,
    projection_config_by_view: dict[str, CapabilityConfigRef] | None = None,
    acquisition_aggregate_step_template: ProtocolStepTemplate | None = None,
    acquisition_aggregate_task_id: str | None = None,
) -> ProtocolTemplate:
    """Fan out a closed unit/group/view roster without carrier semantics."""

    validate_sha256(carrier_fingerprint, field_name="carrier_fingerprint")
    validate_stable_id(source_task_prefix, field_name="source_task_prefix")
    validate_stable_id(projection_task_prefix, field_name="projection_task_prefix")
    if (source_step_template is None) != (projection_step_template is None):
        raise ValueError("acquisition/view expansion requires both adapter step templates")
    if source_step_template is None:
        if source_outputs is None or projection_outputs is None:
            raise ValueError("acquisition/view expansion requires bounded outputs")
        require_sorted_unique_ids(
            source_outputs,
            attribute="output_id",
            field_name="source_outputs",
        )
        require_sorted_unique_ids(
            projection_outputs,
            attribute="output_id",
            field_name="projection_outputs",
        )
        if not source_outputs or not projection_outputs:
            raise ValueError("acquisition/view expansion requires bounded outputs")
    elif source_outputs is not None or projection_outputs is not None:
        raise ValueError("adapter step templates and in-memory output overrides are exclusive")
    by_id = {value.step_id: value for value in template.steps}
    source_id = "linked-source-materialization"
    projection_id = "linked-evidence-projection"
    try:
        source = by_id[source_id]
        projection = by_id[projection_id]
    except KeyError as error:
        raise ValueError("linked protocol lacks its source/projection seam") from error
    if source.dependency_step_ids or projection.dependency_step_ids != (source_id,):
        raise ValueError("linked protocol source/projection topology drifted")

    source_prototype = source if source_step_template is None else source_step_template
    projection_prototype = (
        projection if projection_step_template is None else projection_step_template
    )
    if source_prototype.dependency_step_ids:
        raise ValueError("adapter source step must remain a dependency-free effect")
    if projection_step_template is not None and projection_prototype.dependency_step_ids:
        raise ValueError("adapter projection prototype cannot prebind an acquisition")
    if (acquisition_aggregate_step_template is None) != (acquisition_aggregate_task_id is None):
        raise ValueError("acquisition aggregation requires both a step and task identity")
    if (
        acquisition_aggregate_step_template is not None
        and acquisition_aggregate_step_template.dependency_step_ids
    ):
        raise ValueError("acquisition aggregate prototype cannot prebind source tasks")
    source_output_contract = source_prototype.outputs if source_outputs is None else source_outputs
    projection_output_contract = (
        projection_prototype.outputs if projection_outputs is None else projection_outputs
    )
    if source_config_by_group is not None and set(source_config_by_group) != {
        value.acquisition_group_id for value in acquisition_groups
    }:
        raise ValueError("source config roster differs from acquisition groups")
    if projection_config_by_view is not None and set(projection_config_by_view) != {
        value.view_id for value in nested_views
    }:
        raise ValueError("projection config roster differs from nested views")

    source_steps = tuple(
        replace(
            source_prototype,
            step_id=f"{source_task_prefix}.{group.acquisition_group_id}",
            config=(
                source_prototype.config
                if source_config_by_group is None
                else source_config_by_group[group.acquisition_group_id]
            ),
            outputs=source_output_contract,
            obligation_ids=tuple(
                f"{value}.{group.acquisition_group_id}" for value in source_prototype.obligation_ids
            ),
        )
        for group in acquisition_groups
    )
    source_by_group = {
        group.acquisition_group_id: step
        for group, step in zip(acquisition_groups, source_steps, strict=True)
    }
    projection_steps = tuple(
        replace(
            projection_prototype,
            step_id=f"{projection_task_prefix}.{view.view_id}",
            config=(
                projection_prototype.config
                if projection_config_by_view is None
                else projection_config_by_view[view.view_id]
            ),
            dependency_step_ids=(source_by_group[view.acquisition_group_id].step_id,),
            outputs=projection_output_contract,
            obligation_ids=tuple(
                f"{value}.{view.view_id}" for value in projection_prototype.obligation_ids
            ),
        )
        for view in nested_views
    )
    aggregate_step = (
        None
        if acquisition_aggregate_step_template is None
        else replace(
            acquisition_aggregate_step_template,
            step_id=(
                acquisition_aggregate_task_id
                if acquisition_aggregate_task_id is not None
                else acquisition_aggregate_step_template.step_id
            ),
            dependency_step_ids=tuple(value.step_id for value in source_steps),
        )
    )
    downstream = next(
        (value for value in template.steps if projection_id in value.dependency_step_ids),
        None,
    )
    if downstream is None:
        raise ValueError("linked protocol lacks its post-projection consumer")
    expanded_downstream = replace(
        downstream,
        dependency_step_ids=tuple(
            sorted(
                (
                    *(value.step_id for value in projection_steps),
                    *((aggregate_step.step_id,) if aggregate_step is not None else ()),
                )
            )
        ),
    )
    retained = tuple(
        expanded_downstream if value.step_id == downstream.step_id else value
        for value in template.steps
        if value.step_id not in {source_id, projection_id}
    )
    return ProtocolTemplate(
        template_id=f"{template.template_id}.acquisition-view.{carrier_fingerprint[:16]}",
        template_version=template.template_version,
        steps=tuple(
            sorted(
                (
                    *source_steps,
                    *projection_steps,
                    *((aggregate_step,) if aggregate_step is not None else ()),
                    *retained,
                ),
                key=lambda value: value.step_id,
            )
        ),
        requires_model_set=template.requires_model_set,
        requests_controller=template.requests_controller,
        nonactuating=template.nonactuating,
    )


def rebind_study_template_to_expanded_protocol(
    *,
    study_template: StudyTemplate,
    expanded_protocol: ProtocolTemplate,
    capability_registry: CapabilityRegistry,
) -> StudyTemplate:
    """Rebind one linked candidate graph/coverage to adapter-owned fan-out.

    The adapter has already supplied the complete source/projection step
    contracts.  This function owns only the mechanical candidate topology:
    each protocol dependency receives one scientific edge for every producer
    output accepted by the consumer capability.  No scientific result,
    authority or provider is constructed here.
    """

    original_protocol = study_template.protocol
    if not expanded_protocol.template_id.startswith(f"{original_protocol.template_id}."):
        raise ValueError("expanded protocol does not derive from the selected template")
    original_nodes = {value.node_id: value for value in study_template.graph.nodes}
    nodes = tuple(
        sorted(
            (
                CandidateGraphNode(
                    node_id=step.step_id,
                    stage=step.stage,
                    capability_key=step.capability_key,
                    capability_version=step.capability_version,
                    implementation_sha256=capability_registry.resolve(
                        step.capability_key,
                        step.capability_version,
                    ).implementation_sha256,
                    protocol_step_sha256=step.fingerprint(),
                    obligation_ids=step.obligation_ids,
                    outcome_access=step.requested_outcome_access,
                    visibility_ceiling=step.visibility_ceiling,
                    resource_budget=step.resource_budget,
                    terminal_condition_id=(
                        original_nodes[step.step_id].terminal_condition_id
                        if step.step_id in original_nodes
                        else None
                    ),
                )
                for step in expanded_protocol.steps
            ),
            key=lambda value: value.node_id,
        )
    )
    expanded_step_ids = {step.step_id for step in expanded_protocol.steps}
    original_external_edges = tuple(
        value
        for value in study_template.graph.edges
        if value.producer_node_id is None and value.consumer_node_id in expanded_step_ids
    )
    source_id = "linked-source-materialization"
    original_source = next(
        (value for value in original_protocol.steps if value.step_id == source_id),
        None,
    )
    replacement_source_steps = (
        ()
        if original_source is None or source_id in expanded_step_ids
        else tuple(
            value
            for value in expanded_protocol.steps
            if value.stage is original_source.stage and not value.dependency_step_ids
        )
    )
    source_external_edges = tuple(
        value
        for value in study_template.graph.edges
        if value.producer_node_id is None and value.consumer_node_id == source_id
    )
    expanded_source_external_edges = tuple(
        replace(
            edge,
            edge_id=f"{edge.edge_id}.{step.step_id}",
            consumer_node_id=step.step_id,
            consumer_input_id=f"{edge.consumer_input_id}.{step.step_id}",
        )
        for step in replacement_source_steps
        for edge in source_external_edges
    )
    steps = {value.step_id: value for value in expanded_protocol.steps}
    edges: list[CandidateGraphEdge] = [
        *original_external_edges,
        *expanded_source_external_edges,
    ]
    for child in expanded_protocol.steps:
        child_manifest = capability_registry.resolve(
            child.capability_key,
            child.capability_version,
        )
        for parent_id in child.dependency_step_ids:
            parent = steps[parent_id]
            compatible_outputs = tuple(
                value
                for value in parent.outputs
                if value.payload_schema in child_manifest.input_schema_ids
            )
            if not compatible_outputs:
                raise ValueError(
                    "expanded protocol dependency has no schema-compatible scientific edge: "
                    f"{parent_id}->{child.step_id}"
                )
            for output in compatible_outputs:
                edge_id = f"edge.{parent_id}.{output.output_id}.{child.step_id}"
                edges.append(
                    CandidateGraphEdge(
                        edge_id=edge_id,
                        producer_node_id=parent_id,
                        producer_output_id=output.output_id,
                        external_input_id=None,
                        consumer_node_id=child.step_id,
                        consumer_input_id=f"input-{parent_id}-{output.output_id}",
                        scientific_role=ScientificInputRole.OUTCOME,
                        logical_artifact_id=f"artifact.{parent_id}.{output.output_id}",
                        payload_schema=output.payload_schema,
                        media_type=output.media_type,
                        maximum_size_bytes=parent.resource_budget.output_bytes,
                        outcome_access=child.requested_outcome_access,
                        visibility_ceiling=child.visibility_ceiling,
                        barrier=child.barrier,
                    )
                )
    graph = CandidateScientificGraph(
        graph_id=(
            f"{study_template.graph.graph_id}.acquisition-view."
            f"{expanded_protocol.fingerprint()[:16]}"
        ),
        external_inputs=study_template.graph.external_inputs,
        nodes=nodes,
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )
    incoming = {
        node.node_id: tuple(
            sorted(edge.edge_id for edge in graph.edges if edge.consumer_node_id == node.node_id)
        )
        for node in graph.nodes
    }
    original_coverage = {
        value.obligation_id: value for value in study_template.coverage.bindings
    }
    fallback = next(iter(study_template.coverage.bindings))
    bindings = tuple(
        sorted(
            (
                ObligationCoverageBinding(
                    obligation_id=obligation_id,
                    proof_owner_node_id=step.step_id,
                    required_output_id=(
                        original_coverage.get(obligation_id, fallback).required_output_id
                        if original_coverage.get(obligation_id, fallback).required_output_id
                        in {value.output_id for value in step.outputs}
                        else step.outputs[0].output_id
                    ),
                    contributor_edge_ids=incoming[step.step_id],
                )
                for step in expanded_protocol.steps
                for obligation_id in step.obligation_ids
            ),
            key=lambda value: value.obligation_id,
        )
    )
    retained_nonprotocol = tuple(
        replace(
            value,
            contributor_edge_ids=incoming[value.proof_owner_node_id],
        )
        for value in study_template.coverage.bindings
        if value.obligation_id
        not in {
            obligation_id
            for step in original_protocol.steps
            for obligation_id in step.obligation_ids
        }
        and value.proof_owner_node_id in incoming
    )
    coverage = ObligationCoverage(
        coverage_id=(
            f"{study_template.coverage.coverage_id}.acquisition-view."
            f"{expanded_protocol.fingerprint()[:16]}"
        ),
        bindings=tuple(
            sorted((*bindings, *retained_nonprotocol), key=lambda value: value.obligation_id)
        ),
    )
    return StudyTemplate(
        template_key=study_template.template_key,
        template_version=study_template.template_version,
        protocol=expanded_protocol,
        graph=graph,
        coverage=coverage,
    )


@dataclass(frozen=True, slots=True)
class LinkedCampaignProvider:
    """Static owner delegation and condition accounting; never a task runner."""

    profile: LinkedCampaignProfile

    def owner(self, role: LinkedCampaignOwnerRole) -> LinkedCampaignOwnerBinding:
        return next(value for value in self.profile.owners if value.role is role)

    def resolve_conditional_child(
        self,
        resolver: ConditionalChildResolver,
        *,
        parent_compilation: StudyCompilationReport,
        parent_record: CanonicalRecord,
    ) -> ConditionalChildResolution:
        """Delegate exact parent substitution to the existing frozen resolver."""

        return resolver.resolve(
            parent_compilation=parent_compilation,
            parent_record=parent_record,
        )

    def route(self, state: LinkedCampaignGateState) -> LinkedCampaignRouteDisposition:
        supported_until: LinkedCampaignOwnerRole | None
        terminal_disposition: LinkedCampaignDisposition | None = None
        terminal_reason: str | None = None
        if state.parent_disposition is LinkedCampaignParentDisposition.ZERO_LAW:
            supported_until = LinkedCampaignOwnerRole.BATCH_ATLAS
            terminal_disposition = LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE
            terminal_reason = "ZERO_SUPPORTED_LAWS"
        elif state.parent_disposition is LinkedCampaignParentDisposition.PARTIAL_UNEVALUABLE:
            supported_until = LinkedCampaignOwnerRole.SOLE_LAW_FINALIZER
            terminal_disposition = LinkedCampaignDisposition.UNEVALUABLE
            terminal_reason = "LAW_BATCH_PARTIAL_UNEVALUABLE"
        elif not state.admission_corpus_complete:
            supported_until = LinkedCampaignOwnerRole.ADMISSION_EVIDENCE_PRODUCER
            terminal_disposition = LinkedCampaignDisposition.UNEVALUABLE
            terminal_reason = "ADMISSION_CORPUS_INCOMPLETE"
        elif not state.admission_reachable:
            supported_until = LinkedCampaignOwnerRole.REACHABILITY
            terminal_disposition = LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE
            terminal_reason = "NO_ADMITTED_REACHABLE_CELL"
        elif not state.measured_hold_supported:
            supported_until = LinkedCampaignOwnerRole.PROGRAMME_AUTHOR
            terminal_disposition = LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE
            terminal_reason = "MEASURED_HOLD_UNAVAILABLE"
        elif not state.execution_authorized:
            supported_until = LinkedCampaignOwnerRole.SOLE_CONTROLLER_COMPILER
            terminal_disposition = LinkedCampaignDisposition.AUTHORITY_REQUIRED
            terminal_reason = "EXECUTION_AUTHORITY_REQUIRED"
        elif state.controller_use_qualification is LinkedCampaignControllerUseQualification.FAIL:
            supported_until = LinkedCampaignOwnerRole.CONTROLLER_USE_ROUTE_QUALIFIER
            terminal_disposition = LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE
            terminal_reason = "CONTROLLER_USE_ROUTE_QUALIFICATION_FAILED"
        elif state.controller_use_qualification is LinkedCampaignControllerUseQualification.NOT_REACHED:
            supported_until = LinkedCampaignOwnerRole.CONTROLLER_USE_ROUTE_QUALIFIER
            terminal_disposition = LinkedCampaignDisposition.UNEVALUABLE
            terminal_reason = "CONTROLLER_USE_ROUTE_QUALIFICATION_NOT_REACHED"
        else:
            supported_until = None

        dispositions: list[LinkedCampaignRoleDisposition] = []
        terminal_seen = False
        for role in _OWNER_ORDER:
            if supported_until is None:
                disposition = LinkedCampaignDisposition.SUPPORTED
                reasons: tuple[str, ...] = ()
            elif terminal_seen:
                disposition = LinkedCampaignDisposition.CONDITION_FALSE
                reasons = (f"UPSTREAM_{terminal_reason}",)
            elif role is supported_until:
                if terminal_disposition is None or terminal_reason is None:
                    raise AssertionError("linked route terminal is incomplete")
                disposition = terminal_disposition
                reasons = (terminal_reason,)
                terminal_seen = True
            else:
                disposition = LinkedCampaignDisposition.SUPPORTED
                reasons = ()
            dispositions.append(
                LinkedCampaignRoleDisposition(
                    disposition_id=f"route-disposition.{_stable_role(role)}",
                    role=role,
                    disposition=disposition,
                    reason_codes=reasons,
                )
            )
        return LinkedCampaignRouteDisposition(
            route_id=f"linked-route.{state.state_id}",
            profile=ObjectIdentity.from_record(self.profile.profile_id, self.profile),
            roles=tuple(sorted(dispositions, key=lambda value: value.disposition_id)),
        )


def resolve_complete_unit_reserves(
    group: CompleteUnitReserveGroup,
    *,
    unavailable_primary_unit_ids: tuple[str, ...],
) -> ReserveSubstitutionReceipt:
    require_sorted_unique_strings(
        unavailable_primary_unit_ids,
        field_name="unavailable_primary_unit_ids",
    )
    if not set(unavailable_primary_unit_ids).issubset(group.primary_physical_unit_ids):
        raise ValueError("reserve request names a non-primary physical unit")
    if len(unavailable_primary_unit_ids) > group.maximum_substitutions:
        raise ValueError("reserve request exceeds its predeclared substitution bound")
    ordered_reserves = tuple(sorted(group.reserves, key=lambda value: value.rank))
    substitutions = tuple(
        ReserveSubstitution(
            substitution_id=f"reserve-substitution.{primary}",
            unavailable_primary_unit_id=primary,
            reserve_physical_unit_id=reserve.physical_independent_unit_id,
            reserve_rank=reserve.rank,
        )
        for primary, reserve in zip(
            unavailable_primary_unit_ids,
            ordered_reserves,
            strict=False,
        )
    )
    if len(substitutions) != len(unavailable_primary_unit_ids):
        raise ValueError("reserve roster cannot satisfy the complete-unit substitution")
    final_units = tuple(
        sorted(
            (set(group.primary_physical_unit_ids) - set(unavailable_primary_unit_ids))
            | {value.reserve_physical_unit_id for value in substitutions}
        )
    )
    return ReserveSubstitutionReceipt(
        receipt_id=f"reserve-receipt.{group.group_id}",
        reserve_group=ObjectIdentity.from_record(group.group_id, group),
        substitutions=tuple(sorted(substitutions, key=lambda value: value.substitution_id)),
        final_physical_unit_ids=final_units,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


__all__ = [
    "LinkedCampaignDisposition",
    'LinkedCampaignGateState',
    "LinkedCampaignControllerUseQualification",
    "LinkedCampaignParentDisposition",
    'LinkedCampaignProvider',
    'LinkedCampaignRoleDisposition',
    'LinkedCampaignRouteDisposition',
    'LinkedCampaignStageEnvelope',
    'ReserveSubstitutionReceipt',
    'ReserveSubstitution',
    "build_linked_campaign_protocol",
    'expand_acquisition_views',
    'expand_linked_study_acquisition_views',
    'rebind_study_template_to_expanded_protocol',
    "resolve_complete_unit_reserves",
]
