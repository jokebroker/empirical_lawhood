"""Pure deterministic compilation from strict drafts to authority-pending candidates."""

from __future__ import annotations

from empirical_lawhood.kernel.retrospective_prediction_experiments import RetrospectiveExperimentSpec
from empirical_lawhood.planning.study_authoring import RetrospectiveDesignOrigin, RetrospectiveStudyDraft
from empirical_lawhood.planning.experiment_entry import RetrospectiveAuthoringBase, RetrospectiveAuthoringPackage, RetrospectiveEntryPackage

from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import (
    ExperimentSpec,
    validate_experiment_against_system,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.planning.study_authoring import CapabilitySelection, ConditionalChildRequest, DesignInputRecord, DesignInputRole, DesignOrigin, MaterializationQualificationReceipt, StudyDraft, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.planning.experiment_entry import StudyDefinition, ExecutableStudyDefinition, validate_experiment_entry
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
    FormalMethodCatalog,
    FormalMethodRole,
    derive_formal_gap_applicability,
)
from empirical_lawhood.planning.formal_gaps import FormalGapCoverageDisposition
from empirical_lawhood.planning.campaigns import CampaignSpec
from empirical_lawhood.kernel.systems import SystemSpec

from .capabilities import (
    CapabilityKind,
    CapabilityRegistry,
)
from .plans import (
    BarrierKind,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificInputRole,
    ScientificStage,
)
from .study_extensions import StudyExtensionMaterializationReceipt


class ContentIdentityPolicy(StrEnum):
    EXACT_SHA256 = "EXACT_SHA256"
    RECEIPT_BOUND = "RECEIPT_BOUND"
    PARENT_RECEIPT_SUBSTITUTION = "PARENT_RECEIPT_SUBSTITUTION"


class CandidateDiagnosticClass(StrEnum):
    FATAL = "FATAL"
    UNRESOLVED_READINESS = "UNRESOLVED_READINESS"
    AUTHORITY_GATE = "AUTHORITY_GATE"
    SUCCESS = "SUCCESS"


class CandidateDiagnosticCode(StrEnum):
    SCHEMA_INVALID = "SCHEMA_INVALID"
    SCIENTIFIC_SPEC_INCOMPLETE = "SCIENTIFIC_SPEC_INCOMPLETE"
    SCIENTIFIC_OBLIGATION_UNCOVERED = "SCIENTIFIC_OBLIGATION_UNCOVERED"
    CAPABILITY_CONTRACT_MISMATCH = "CAPABILITY_CONTRACT_MISMATCH"
    OUTCOME_SEPARATION_REQUIRED = "OUTCOME_SEPARATION_REQUIRED"
    RESOURCE_UNBOUNDED = "RESOURCE_UNBOUNDED"
    SOURCE_UNRESOLVED = "SOURCE_UNRESOLVED"
    SOURCE_QUALIFICATION_REQUIRED = "SOURCE_QUALIFICATION_REQUIRED"
    SOURCE_ACQUISITION_AUTHORITY_REQUIRED = "SOURCE_ACQUISITION_AUTHORITY_REQUIRED"
    CAPABILITY_REQUIRED = "CAPABILITY_REQUIRED"
    CUSTODY_PUBLICATION_AUTHORITY_REQUIRED = "CUSTODY_PUBLICATION_AUTHORITY_REQUIRED"
    SCIENTIFIC_APPROVAL_AUTHORITY_REQUIRED = "SCIENTIFIC_APPROVAL_AUTHORITY_REQUIRED"
    EXPERIMENT_EXECUTION_AUTHORITY_REQUIRED = "EXPERIMENT_EXECUTION_AUTHORITY_REQUIRED"
    OUTCOME_REVEAL_AUTHORITY_REQUIRED = "OUTCOME_REVEAL_AUTHORITY_REQUIRED"
    FORMAL_GAP_COVERAGE_REQUIRED = "FORMAL_GAP_COVERAGE_REQUIRED"
    FORMAL_GAP_APPLICABILITY_MISMATCH = "FORMAL_GAP_APPLICABILITY_MISMATCH"
    FORMAL_GAP_METHOD_REQUIRED = "FORMAL_GAP_METHOD_REQUIRED"
    FORMAL_GAP_BINDING_MISMATCH = "FORMAL_GAP_BINDING_MISMATCH"
    FORMAL_GAP_FEASIBLE_OPERAND_OMITTED = "FORMAL_GAP_FEASIBLE_OPERAND_OMITTED"
    FORMAL_GAP_GRAPH_BINDING_INVALID = "FORMAL_GAP_GRAPH_BINDING_INVALID"
    FORMAL_GAP_CEILING_EXCEEDED = "FORMAL_GAP_CEILING_EXCEEDED"
    FORMAL_GAP_PANEL_INCOMPLETE = "FORMAL_GAP_PANEL_INCOMPLETE"
    READY_TO_ISSUE = "READY_TO_ISSUE"


class CandidateCompilationDisposition(StrEnum):
    INVALID_DRAFT = "INVALID_DRAFT"
    UNRESOLVED_READINESS = "UNRESOLVED_READINESS"
    COMPILED_AUTHORITY_PENDING = "COMPILED_AUTHORITY_PENDING"


class CandidateAxisState(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"
    EXPECTED_AUTHORITY_GATE = "EXPECTED_AUTHORITY_GATE"


@dataclass(frozen=True, slots=True)
class AuthoringMaterializationIdentity(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/authoring-materialization-identity'

    media_type: str
    byte_count: int
    raw_materialization_sha256: str

    def __post_init__(self) -> None:
        if self.media_type not in {"application/json", "application/yaml"}:
            raise ValueError("unsupported programme-draft media type")
        if self.byte_count <= 0:
            raise ValueError("programme-draft materialization must be nonempty")
        validate_sha256(
            self.raw_materialization_sha256,
            field_name="raw_materialization_sha256",
        )


@dataclass(frozen=True, slots=True)
class CandidateGraphExternalInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-graph-external-input'

    input_id: str
    scientific_role: ScientificInputRole
    logical_artifact_id: str
    content_identity_policy: ContentIdentityPolicy
    expected_content_sha256: str | None
    payload_schema: str
    media_type: str
    maximum_size_bytes: int
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        validate_stable_id(
            self.logical_artifact_id,
            field_name="logical_artifact_id",
        )
        validate_schema(self.payload_schema)
        validate_nonempty(self.media_type, field_name="media_type")
        if self.maximum_size_bytes <= 0:
            raise ValueError("external input requires a positive size bound")
        if self.content_identity_policy is ContentIdentityPolicy.EXACT_SHA256:
            if self.expected_content_sha256 is None:
                raise ValueError("exact external input requires a content digest")
            validate_sha256(
                self.expected_content_sha256,
                field_name="expected_content_sha256",
            )
        elif self.expected_content_sha256 is not None:
            validate_sha256(
                self.expected_content_sha256,
                field_name="expected_content_sha256",
            )
        if (
            self.content_identity_policy is ContentIdentityPolicy.PARENT_RECEIPT_SUBSTITUTION
            and self.scientific_role is not ScientificInputRole.PARENT_RECEIPT
        ):
            raise ValueError("parent-receipt substitution requires parent-receipt role")


@dataclass(frozen=True, slots=True)
class CandidateGraphNode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-graph-node'

    node_id: str
    stage: ScientificStage
    capability_key: str
    capability_version: str
    implementation_sha256: str
    protocol_step_sha256: str
    obligation_ids: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    resource_budget: ResourceBudget
    terminal_condition_id: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("node_id", self.node_id),
            ("capability_key", self.capability_key),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.capability_version)
        validate_sha256(
            self.implementation_sha256,
            field_name="implementation_sha256",
        )
        validate_sha256(
            self.protocol_step_sha256,
            field_name="protocol_step_sha256",
        )
        require_sorted_unique_strings(
            self.obligation_ids,
            field_name="obligation_ids",
            allow_empty=False,
        )
        if self.terminal_condition_id is not None:
            validate_stable_id(
                self.terminal_condition_id,
                field_name="terminal_condition_id",
            )


@dataclass(frozen=True, slots=True)
class CandidateGraphEdge(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-graph-edge'

    edge_id: str
    producer_node_id: str | None
    producer_output_id: str | None
    external_input_id: str | None
    consumer_node_id: str
    consumer_input_id: str
    scientific_role: ScientificInputRole
    logical_artifact_id: str
    payload_schema: str
    media_type: str
    maximum_size_bytes: int
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    barrier: BarrierKind

    def __post_init__(self) -> None:
        for name, value in (
            ("edge_id", self.edge_id),
            ("consumer_node_id", self.consumer_node_id),
            ("consumer_input_id", self.consumer_input_id),
            ("logical_artifact_id", self.logical_artifact_id),
        ):
            validate_stable_id(value, field_name=name)
        internal = self.producer_node_id is not None or self.producer_output_id is not None
        external = self.external_input_id is not None
        if internal == external:
            raise ValueError("edge must bind exactly one internal or external producer")
        if internal:
            if self.producer_node_id is None or self.producer_output_id is None:
                raise ValueError("internal edge requires node and output IDs")
            validate_stable_id(
                self.producer_node_id,
                field_name="producer_node_id",
            )
            validate_stable_id(
                self.producer_output_id,
                field_name="producer_output_id",
            )
        else:
            assert self.external_input_id is not None
            validate_stable_id(
                self.external_input_id,
                field_name="external_input_id",
            )
        validate_schema(self.payload_schema)
        validate_nonempty(self.media_type, field_name="media_type")
        if self.maximum_size_bytes <= 0:
            raise ValueError("scientific edge requires a positive size bound")


@dataclass(frozen=True, slots=True)
class CandidateScientificGraph(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-scientific-graph'

    graph_id: str
    external_inputs: tuple[CandidateGraphExternalInput, ...]
    nodes: tuple[CandidateGraphNode, ...]
    edges: tuple[CandidateGraphEdge, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.graph_id, field_name="graph_id")
        require_sorted_unique_ids(
            self.external_inputs,
            attribute="input_id",
            field_name="external_inputs",
        )
        require_sorted_unique_ids(
            self.nodes,
            attribute="node_id",
            field_name="nodes",
        )
        require_sorted_unique_ids(
            self.edges,
            attribute="edge_id",
            field_name="edges",
        )
        if not self.nodes:
            raise ValueError("candidate scientific graph requires nodes")
        self._validate_edges()
        self._validate_acyclicity()

    def _validate_edges(self) -> None:
        node_ids = {node.node_id for node in self.nodes}
        external_inputs = {value.input_id: value for value in self.external_inputs}
        consumers: set[tuple[str, str]] = set()
        for edge in self.edges:
            if edge.consumer_node_id not in node_ids:
                raise ValueError("scientific edge consumes at an unknown node")
            consumer = (edge.consumer_node_id, edge.consumer_input_id)
            if consumer in consumers:
                raise ValueError("scientific consumer input has multiple producers")
            consumers.add(consumer)
            if edge.producer_node_id is not None:
                if edge.producer_node_id not in node_ids:
                    raise ValueError("scientific edge originates at an unknown node")
                if edge.producer_node_id == edge.consumer_node_id:
                    raise ValueError("scientific edge cannot be a self-loop")
            else:
                external = external_inputs.get(edge.external_input_id or "")
                if external is None:
                    raise ValueError("scientific edge references an unknown external input")
                if (
                    edge.scientific_role is not external.scientific_role
                    or edge.logical_artifact_id != external.logical_artifact_id
                    or edge.payload_schema != external.payload_schema
                    or edge.media_type != external.media_type
                    or edge.maximum_size_bytes > external.maximum_size_bytes
                    or edge.outcome_access is not external.outcome_access
                    or edge.visibility_ceiling is not external.visibility_ceiling
                ):
                    raise ValueError(
                        "scientific edge differs from its exact external-input contract"
                    )

    def _validate_acyclicity(self) -> None:
        parents: dict[str, set[str]] = {node.node_id: set() for node in self.nodes}
        for edge in self.edges:
            if edge.producer_node_id is not None:
                parents[edge.consumer_node_id].add(edge.producer_node_id)
        for start in parents:
            pending = [start]
            path: set[str] = set()
            completed: set[str] = set()
            while pending:
                current = pending[-1]
                if current in completed:
                    pending.pop()
                    continue
                if current in path:
                    path.remove(current)
                    completed.add(current)
                    pending.pop()
                    continue
                path.add(current)
                for parent in parents[current]:
                    if parent in path:
                        raise ValueError("candidate scientific graph contains a cycle")
                    if parent not in completed:
                        pending.append(parent)


@dataclass(frozen=True, slots=True)
class ObligationCoverageBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/obligation-coverage-binding'

    obligation_id: str
    proof_owner_node_id: str
    required_output_id: str
    contributor_edge_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("obligation_id", self.obligation_id),
            ("proof_owner_node_id", self.proof_owner_node_id),
            ("required_output_id", self.required_output_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.contributor_edge_ids,
            field_name="contributor_edge_ids",
        )


@dataclass(frozen=True, slots=True)
class ObligationCoverage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/obligation-coverage'

    coverage_id: str
    bindings: tuple[ObligationCoverageBinding, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.coverage_id, field_name="coverage_id")
        require_sorted_unique_ids(
            self.bindings,
            attribute="obligation_id",
            field_name="bindings",
        )
        if not self.bindings:
            raise ValueError("obligation coverage cannot be empty")


@dataclass(frozen=True, slots=True)
class StudyTemplate(CanonicalRecord):
    """Static compiler-owned protocol, exact graph and coverage recipe."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/study-template'

    template_key: str
    template_version: str
    protocol: ProtocolTemplate
    graph: CandidateScientificGraph
    coverage: ObligationCoverage

    def __post_init__(self) -> None:
        validate_stable_id(self.template_key, field_name="template_key")
        validate_semantic_version(self.template_version)
        self._validate_protocol_graph_parity()
        self._validate_coverage_owners()

    def _validate_protocol_graph_parity(self) -> None:
        steps = {step.step_id: step for step in self.protocol.steps}
        nodes = {node.node_id: node for node in self.graph.nodes}
        if set(steps) != set(nodes):
            raise ValueError("programme template graph and protocol nodes differ")
        for node_id, node in nodes.items():
            step = steps[node_id]
            if (
                node.stage is not step.stage
                or node.capability_key != step.capability_key
                or node.capability_version != step.capability_version
                or node.protocol_step_sha256 != step.fingerprint()
                or node.obligation_ids != step.obligation_ids
                or node.outcome_access is not step.requested_outcome_access
                or node.visibility_ceiling is not step.visibility_ceiling
                or node.resource_budget != step.resource_budget
            ):
                raise ValueError("programme template node differs from protocol step")
            observed_parents = tuple(
                sorted(
                    {
                        edge.producer_node_id
                        for edge in self.graph.edges
                        if edge.consumer_node_id == node_id and edge.producer_node_id is not None
                    }
                )
            )
            if observed_parents != step.dependency_step_ids:
                raise ValueError("exact scientific edges differ from protocol dependencies")
        for edge in self.graph.edges:
            if edge.producer_node_id is None:
                continue
            producer = steps[edge.producer_node_id]
            output = next(
                (value for value in producer.outputs if value.output_id == edge.producer_output_id),
                None,
            )
            if output is None:
                raise ValueError("scientific edge names an unknown producer output")
            if edge.payload_schema != output.payload_schema or edge.media_type != output.media_type:
                raise ValueError("scientific edge payload differs from producer output")
            producer_node = nodes[edge.producer_node_id]
            if not edge.visibility_ceiling.is_at_least_as_restrictive_as(
                producer_node.visibility_ceiling
            ):
                raise ValueError("scientific edge lowers producer visibility")

    def _validate_coverage_owners(self) -> None:
        steps = {step.step_id: step for step in self.protocol.steps}
        edges = {edge.edge_id for edge in self.graph.edges}
        for binding in self.coverage.bindings:
            step = steps.get(binding.proof_owner_node_id)
            if step is None:
                raise ValueError("obligation proof owner is not a protocol node")
            if binding.required_output_id not in {output.output_id for output in step.outputs}:
                raise ValueError("obligation proof output is not owned by its node")
            if not set(binding.contributor_edge_ids).issubset(edges):
                raise ValueError("obligation coverage names an unknown contributor edge")
        protocol_obligations = {
            obligation_id for step in self.protocol.steps for obligation_id in step.obligation_ids
        }
        covered = {value.obligation_id for value in self.coverage.bindings}
        if not protocol_obligations.issubset(covered):
            raise ValueError("programme template leaves protocol obligations uncovered")


@dataclass(frozen=True, slots=True)
class CandidateCompilationContext(CanonicalRecord):
    """Closed injected registry/context; never document-selected dispatch."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-compilation-context'

    context_id: str
    registry: CapabilityRegistry
    templates: tuple[StudyTemplate, ...]
    qualifications: tuple[MaterializationQualificationReceipt, ...]
    known_design_inputs: tuple[DesignInputRecord, ...]
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.context_id, field_name="context_id")
        require_sorted_unique_ids(
            self.templates,
            attribute="template_key",
            field_name="templates",
        )
        require_sorted_unique_ids(
            self.qualifications,
            attribute="receipt_id",
            field_name="qualifications",
        )
        require_sorted_unique_ids(
            self.known_design_inputs,
            attribute="input_id",
            field_name="known_design_inputs",
        )
        validate_sha256(
            self.implementation_sha256,
            field_name="implementation_sha256",
        )

    def template(self, template_key: str) -> StudyTemplate | None:
        return next(
            (value for value in self.templates if value.template_key == template_key),
            None,
        )


@dataclass(frozen=True, slots=True)
class StandardCandidateCompilationContext(CanonicalRecord):
    """Additive context for mandatory formal-gap compilation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/standard-candidate-compilation-context'

    context_id: str
    base: CandidateCompilationContext
    formal_methods: FormalMethodCatalog
    source_inventories: tuple[FormalGapSourceCapabilityInventory, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.context_id, field_name="context_id")
        require_sorted_unique_ids(
            self.source_inventories,
            attribute="denominator_id",
            field_name="source_inventories",
        )
        if not self.source_inventories:
            raise ValueError("standard candidate context requires source inventories")

    def source_inventory(
        self,
        denominator_id: str,
    ) -> FormalGapSourceCapabilityInventory | None:
        return next(
            (value for value in self.source_inventories if value.denominator_id == denominator_id),
            None,
        )


@dataclass(frozen=True, slots=True)
class FrozenConditionalChild(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/frozen-conditional-child'

    request: ConditionalChildRequest
    protocol: ProtocolTemplate
    scientific_graph: CandidateScientificGraph
    obligation_coverage: ObligationCoverage
    scientific_template_sha256: str
    allowed_instantiation_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_sha256(
            self.scientific_template_sha256,
            field_name="scientific_template_sha256",
        )
        require_sorted_unique_strings(
            self.allowed_instantiation_fields,
            field_name="allowed_instantiation_fields",
            allow_empty=False,
        )
        expected = (
            "authority_id",
            "issue_id",
            "parent_receipt",
            "run_id",
            "task_id",
        )
        if self.allowed_instantiation_fields != expected:
            raise ValueError("conditional child permits undeclared scientific edits")
        matches = [
            value
            for value in self.scientific_graph.external_inputs
            if value.input_id == self.request.parent_receipt_input_id
            and value.scientific_role is ScientificInputRole.PARENT_RECEIPT
            and value.content_identity_policy is ContentIdentityPolicy.PARENT_RECEIPT_SUBSTITUTION
        ]
        if len(matches) != 1:
            raise ValueError("conditional child lacks its sole parent-receipt slot")


@dataclass(frozen=True, slots=True)
class CandidateDiagnostic(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-diagnostic'

    diagnostic_id: str
    diagnostic_class: CandidateDiagnosticClass
    code: CandidateDiagnosticCode
    field_path: str
    message: str

    def __post_init__(self) -> None:
        validate_stable_id(self.diagnostic_id, field_name="diagnostic_id")
        validate_nonempty(self.field_path, field_name="field_path")
        validate_nonempty(self.message, field_name="message")


@dataclass(frozen=True, slots=True)
class DraftStudyCandidate(CanonicalRecord):
    """Nonauthoritative deterministic candidate; never executable or approved."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/draft-study-candidate'

    candidate_id: str
    disposition: CandidateCompilationDisposition
    authoring_materialization: AuthoringMaterializationIdentity
    semantic_config_sha256: str
    resolved_context_sha256: str
    implementation_sha256: str
    system: SystemSpec
    experiment: ExperimentSpec
    campaign: CampaignSpec
    protocol: ProtocolTemplate
    scientific_graph: CandidateScientificGraph
    obligation_coverage: ObligationCoverage
    design_origin: DesignOrigin
    design_input_ids: tuple[str, ...]
    capability_locks: tuple[CapabilitySelection, ...]
    source_locks: tuple[SourceMaterializationRef, ...]
    source_qualification_receipts: tuple[ObjectIdentity, ...]
    resource_ceiling: ResourceBudget
    conditional_successor: FrozenConditionalChild | None
    design_validity_state: CandidateAxisState
    source_readiness_state: CandidateAxisState
    freeze_readiness_state: CandidateAxisState
    authority_state: CandidateAxisState
    expected_authority_gates: tuple[str, ...]
    required_issue_inputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveStudyCandidate):
            if type(self.experiment) is not ExperimentSpec:
                raise ValueError("ProgrammeCandidate requires its original experiment schema")
            if type(self.design_origin) is not DesignOrigin:
                raise ValueError("ProgrammeCandidate requires its original design_origin schema")
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        if self.disposition is not CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING:
            raise ValueError("candidate must remain authority pending")
        for name, value in (
            ("semantic_config_sha256", self.semantic_config_sha256),
            ("resolved_context_sha256", self.resolved_context_sha256),
            ("implementation_sha256", self.implementation_sha256),
        ):
            validate_sha256(value, field_name=name)
        require_sorted_unique_strings(
            self.design_input_ids,
            field_name="design_input_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.capability_locks,
            attribute="selection_id",
            field_name="capability_locks",
        )
        require_sorted_unique_ids(
            self.source_locks,
            attribute="source_id",
            field_name="source_locks",
        )
        require_sorted_unique_ids(
            self.source_qualification_receipts,
            attribute="object_id",
            field_name="source_qualification_receipts",
        )
        if (
            self.design_validity_state,
            self.source_readiness_state,
            self.freeze_readiness_state,
            self.authority_state,
        ) != (
            CandidateAxisState.READY,
            CandidateAxisState.READY,
            CandidateAxisState.READY,
            CandidateAxisState.EXPECTED_AUTHORITY_GATE,
        ):
            raise ValueError("compiled candidate readiness axes are inconsistent")
        require_sorted_unique_strings(
            self.expected_authority_gates,
            field_name="expected_authority_gates",
            allow_empty=False,
        )
        if self.required_issue_inputs != _REQUIRED_ISSUE_INPUTS:
            raise ValueError("candidate issue requirements differ from the closed contract")


@dataclass(frozen=True, slots=True)
class StudyCandidate(CanonicalRecord):
    """Standard claim-bearing candidate bound to formal entry evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/study-candidate'

    candidate_id: str
    base_candidate: DraftStudyCandidate
    authoring_package: ObjectIdentity
    entry_package: ObjectIdentity
    formal_gap_register: ObjectIdentity
    formal_gap_coverage: ObjectIdentity
    formal_method_catalog: ObjectIdentity
    formal_source_inventory: ObjectIdentity
    standard_context_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveStandardCandidate):
            if type(self.base_candidate) is not DraftStudyCandidate:
                raise ValueError(
                    "StudyCandidate requires its original base_candidate schema"
                )
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        for name, value in (("standard_context_sha256", self.standard_context_sha256),):
            validate_sha256(value, field_name=name)
        if self.authoring_package.object_schema != (
            RetrospectiveAuthoringBase.SCHEMA
            if isinstance(self, RetrospectiveStandardCandidate)
            else StudyDefinition.SCHEMA
        ):
            raise ValueError("standard candidate binds another authoring root")
        if self.entry_package.object_schema != (
            RetrospectiveEntryPackage.SCHEMA
            if isinstance(self, (RetrospectiveStandardCandidate, RetrospectiveStandardReport))
            else 'empirical-lawhood/planning/experiment-entry-package'
        ):
            raise ValueError("standard candidate binds another entry package")
        if self.formal_gap_register.object_schema != ('empirical-lawhood/planning/formal-gap-register'):
            raise ValueError("standard candidate binds another formal register")
        if self.formal_gap_coverage.object_schema != ('empirical-lawhood/planning/formal-gap-coverage'):
            raise ValueError("standard candidate binds another formal coverage")
        if self.formal_method_catalog.object_schema != FormalMethodCatalog.SCHEMA:
            raise ValueError("standard candidate binds another formal method catalog")
        if self.formal_source_inventory.object_schema != FormalGapSourceCapabilityInventory.SCHEMA:
            raise ValueError("standard candidate binds another formal source inventory")


@dataclass(frozen=True, slots=True)
class ExecutableStudyCandidate(CanonicalRecord):
    "Exact additive candidate wrapper retaining the complete StudyCandidate."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-study-candidate'

    candidate_id: str
    base_candidate: StudyCandidate
    authoring_package: ExecutableStudyDefinition
    proposed_extension_set: ObjectIdentity

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveExtensionCandidate):
            if type(self.base_candidate) is not StudyCandidate:
                raise ValueError(
                    "ExecutableStudyCandidate requires its original base_candidate schema"
                )
            if type(self.authoring_package) is not ExecutableStudyDefinition:
                raise ValueError(
                    "ExecutableStudyCandidate requires its original authoring_package schema"
                )
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        if self.base_candidate.authoring_package != ObjectIdentity.from_record(
            self.authoring_package.base.package_id,
            self.authoring_package.base,
        ):
            raise ValueError("executable study candidate wrapper changes its base authoring package")
        if self.proposed_extension_set != ObjectIdentity.from_record(
            self.authoring_package.extension_set.extension_set_id,
            self.authoring_package.extension_set,
        ):
            raise ValueError("executable study candidate wrapper binds another proposed extension set")


def bind_standard_candidate_extensions(
    *,
    base_candidate: StudyCandidate,
    authoring_package: ExecutableStudyDefinition,
) -> ExecutableStudyCandidate:
    candidate_record_type: type[ExecutableStudyCandidate] = (
        RetrospectiveExtensionCandidate
        if isinstance(authoring_package, RetrospectiveAuthoringPackage)
        else ExecutableStudyCandidate
    )
    seed = {
        "base_candidate": ObjectIdentity.from_record(
            base_candidate.candidate_id,
            base_candidate,
        ),
        "authoring_package": ObjectIdentity.from_record(
            authoring_package.package_id,
            authoring_package,
        ),
        "extension_set": ObjectIdentity.from_record(
            authoring_package.extension_set.extension_set_id,
            authoring_package.extension_set,
        ),
    }
    return candidate_record_type(
        candidate_id=f"executable-study-candidate.{_digest(seed)[:32]}",
        base_candidate=base_candidate,
        authoring_package=authoring_package,
        proposed_extension_set=seed["extension_set"],
    )


@dataclass(frozen=True, slots=True)
class CandidateCompilationReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-compilation-report'

    draft_id: str
    disposition: CandidateCompilationDisposition
    authoring_materialization: AuthoringMaterializationIdentity
    semantic_config_sha256: str
    resolved_context_sha256: str
    implementation_sha256: str
    system_sha256: str | None
    experiment_sha256: str | None
    campaign_sha256: str | None
    protocol_sha256: str | None
    capability_locks_sha256: str
    source_locks_sha256: str
    resource_ceiling_sha256: str
    capability_lock_ids: tuple[str, ...]
    source_lock_ids: tuple[str, ...]
    compatible_qualification_receipt_ids: tuple[str, ...]
    resource_ceiling: ResourceBudget
    diagnostics: tuple[CandidateDiagnostic, ...]
    candidate: DraftStudyCandidate | None
    provisional_protocol: ProtocolTemplate | None
    provisional_graph: CandidateScientificGraph | None
    provisional_coverage: ObligationCoverage | None
    provisional_graph_sha256: str | None
    provisional_coverage_sha256: str | None
    design_validity_state: CandidateAxisState
    source_readiness_state: CandidateAxisState
    freeze_readiness_state: CandidateAxisState
    authority_state: CandidateAxisState
    expected_authority_gates: tuple[str, ...]
    required_issue_inputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveCandidateReport):
            if self.candidate is not None and type(self.candidate) is not DraftStudyCandidate:
                raise ValueError(
                    "CandidateCompilationReport requires its original candidate schema"
                )
        validate_stable_id(self.draft_id, field_name="draft_id")
        for name, value in (
            ("semantic_config_sha256", self.semantic_config_sha256),
            ("resolved_context_sha256", self.resolved_context_sha256),
            ("implementation_sha256", self.implementation_sha256),
            ("capability_locks_sha256", self.capability_locks_sha256),
            ("source_locks_sha256", self.source_locks_sha256),
            ("resource_ceiling_sha256", self.resource_ceiling_sha256),
        ):
            validate_sha256(value, field_name=name)
        require_sorted_unique_ids(
            self.diagnostics,
            attribute="diagnostic_id",
            field_name="diagnostics",
        )
        for name, values in (
            ("capability_lock_ids", self.capability_lock_ids),
            ("source_lock_ids", self.source_lock_ids),
            (
                "compatible_qualification_receipt_ids",
                self.compatible_qualification_receipt_ids,
            ),
        ):
            require_sorted_unique_strings(values, field_name=name)
        for name, optional_digest in (
            ("system_sha256", self.system_sha256),
            ("experiment_sha256", self.experiment_sha256),
            ("campaign_sha256", self.campaign_sha256),
            ("protocol_sha256", self.protocol_sha256),
            ("provisional_graph_sha256", self.provisional_graph_sha256),
            ("provisional_coverage_sha256", self.provisional_coverage_sha256),
        ):
            if optional_digest is not None:
                validate_sha256(optional_digest, field_name=name)
        if self.resource_ceiling.fingerprint() != self.resource_ceiling_sha256:
            raise ValueError("candidate report resource ceiling digest differs")
        if (
            None if self.provisional_protocol is None else self.provisional_protocol.fingerprint()
        ) != self.protocol_sha256:
            raise ValueError("candidate report protocol digest differs")
        if (
            None if self.provisional_graph is None else self.provisional_graph.fingerprint()
        ) != self.provisional_graph_sha256:
            raise ValueError("candidate report graph digest differs")
        if (
            None if self.provisional_coverage is None else self.provisional_coverage.fingerprint()
        ) != self.provisional_coverage_sha256:
            raise ValueError("candidate report coverage digest differs")
        require_sorted_unique_strings(
            self.expected_authority_gates,
            field_name="expected_authority_gates",
            allow_empty=False,
        )
        if self.required_issue_inputs != _REQUIRED_ISSUE_INPUTS:
            raise ValueError("candidate report issue requirements differ")
        if self.authority_state is not CandidateAxisState.EXPECTED_AUTHORITY_GATE:
            raise ValueError("candidate report cannot grant or omit later authority gates")
        if self.freeze_readiness_state is CandidateAxisState.EXPECTED_AUTHORITY_GATE:
            raise ValueError("freeze readiness is not an authority state")
        if self.disposition is CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING:
            if self.candidate is None:
                raise ValueError("successful candidate report lacks its candidate")
            if (
                self.design_validity_state,
                self.source_readiness_state,
                self.freeze_readiness_state,
            ) != (
                CandidateAxisState.READY,
                CandidateAxisState.READY,
                CandidateAxisState.READY,
            ):
                raise ValueError("successful candidate report has blocked readiness axes")
        elif self.candidate is not None:
            raise ValueError("blocked candidate report cannot contain an issuable identity")


@dataclass(frozen=True, slots=True)
class StudyCompilationReport(CanonicalRecord):
    """Additive public report for the mandatory standard authoring root."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/study-compilation-report'

    report_id: str
    authoring_package: ObjectIdentity
    entry_package: ObjectIdentity
    standard_context_sha256: str
    base_report: CandidateCompilationReport
    candidate: StudyCandidate | None

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveStandardReport):
            if type(self.base_report) is not CandidateCompilationReport:
                raise ValueError(
                    "StudyCompilationReport requires its original base_report schema"
                )
            if (
                self.candidate is not None
                and type(self.candidate) is not StudyCandidate
            ):
                raise ValueError(
                    "StudyCompilationReport requires its original candidate schema"
                )
        validate_stable_id(self.report_id, field_name="report_id")
        validate_sha256(
            self.standard_context_sha256,
            field_name="standard_context_sha256",
        )
        if self.authoring_package.object_schema != (
            RetrospectiveAuthoringBase.SCHEMA
            if isinstance(self, RetrospectiveStandardReport)
            else StudyDefinition.SCHEMA
        ):
            raise ValueError("standard report binds another authoring root")
        if self.entry_package.object_schema != (
            RetrospectiveEntryPackage.SCHEMA
            if isinstance(self, (RetrospectiveStandardCandidate, RetrospectiveStandardReport))
            else 'empirical-lawhood/planning/experiment-entry-package'
        ):
            raise ValueError("standard report binds another entry package")
        if (
            self.base_report.disposition
            is CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING
        ) != (self.candidate is not None):
            raise ValueError("standard report candidate/disposition differs")

    @property
    def disposition(self) -> CandidateCompilationDisposition:
        return self.base_report.disposition

    @property
    def diagnostics(self) -> tuple[CandidateDiagnostic, ...]:
        return self.base_report.diagnostics


@dataclass(frozen=True, slots=True)
class ExecutableStudyCompilationReport(CanonicalRecord):
    "Extension-aware report retaining the complete StudyCompilationReport."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-study-compilation-report'

    report_id: str
    authoring_package: ObjectIdentity
    authoring_materialization: AuthoringMaterializationIdentity
    proposed_extension_set: ObjectIdentity
    extension_materializations: tuple[StudyExtensionMaterializationReceipt, ...]
    base_report: StudyCompilationReport
    candidate: ExecutableStudyCandidate | None

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveExtensionReport):
            if type(self.base_report) is not StudyCompilationReport:
                raise ValueError(
                    "ExecutableStudyCompilationReport requires its original base_report schema"
                )
            if (
                self.candidate is not None
                and type(self.candidate) is not ExecutableStudyCandidate
            ):
                raise ValueError(
                    "ExecutableStudyCompilationReport requires its original candidate schema"
                )
        validate_stable_id(self.report_id, field_name="report_id")
        if self.authoring_package.object_schema != (
            RetrospectiveAuthoringPackage.SCHEMA
            if isinstance(self, RetrospectiveExtensionReport)
            else ExecutableStudyDefinition.SCHEMA
        ):
            raise ValueError("executable study compilation report binds another authoring root")
        if self.proposed_extension_set.object_schema != (
            'empirical-lawhood/planning/proposed-study-extension-set'
        ):
            raise ValueError("executable study compilation report binds another extension set")
        require_sorted_unique_ids(
            self.extension_materializations,
            attribute="extension_id",
            field_name="extension_materializations",
        )
        if (
            self.base_report.disposition
            is CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING
        ) != (self.candidate is not None):
            raise ValueError("executable study compilation report candidate/disposition differs")
        if self.candidate is not None:
            if (
                self.candidate.authoring_package.package_id != self.authoring_package.object_id
                or self.candidate.proposed_extension_set != self.proposed_extension_set
                or tuple(value.extension_id for value in self.extension_materializations)
                != tuple(
                    value.extension_id
                    for value in self.candidate.authoring_package.extension_set.extensions
                )
            ):
                raise ValueError("executable study compilation report changes candidate extension identity")

    @property
    def disposition(self) -> CandidateCompilationDisposition:
        return self.base_report.disposition

    @property
    def diagnostics(self) -> tuple[CandidateDiagnostic, ...]:
        return self.base_report.diagnostics


def bind_standard_candidate_compilation_report(
    *,
    authoring_package: ExecutableStudyDefinition,
    authoring_materialization: AuthoringMaterializationIdentity,
    extension_materializations: tuple[StudyExtensionMaterializationReceipt, ...],
    base_report: StudyCompilationReport,
) -> ExecutableStudyCompilationReport:
    """Bind verified extension bytes only when the unchanged base compiled."""

    compilation_report_record_type: type[ExecutableStudyCompilationReport] = (
        RetrospectiveExtensionReport
        if isinstance(authoring_package, RetrospectiveAuthoringPackage)
        else ExecutableStudyCompilationReport
    )
    candidate = (
        None
        if base_report.candidate is None
        else bind_standard_candidate_extensions(
            base_candidate=base_report.candidate,
            authoring_package=authoring_package,
        )
    )
    authoring_identity = ObjectIdentity.from_record(
        authoring_package.package_id,
        authoring_package,
    )
    seed = {
        "authoring_package": authoring_identity,
        "authoring_materialization": authoring_materialization,
        "extension_materializations": tuple(
            ObjectIdentity.from_record(value.receipt_id, value)
            for value in extension_materializations
        ),
        "base_report": ObjectIdentity.from_record(base_report.report_id, base_report),
        "candidate": (
            None
            if candidate is None
            else ObjectIdentity.from_record(candidate.candidate_id, candidate)
        ),
    }
    return compilation_report_record_type(
        report_id=f"executable-study-compilation-report.{_digest(seed)[:32]}",
        authoring_package=authoring_identity,
        authoring_materialization=authoring_materialization,
        proposed_extension_set=ObjectIdentity.from_record(
            authoring_package.extension_set.extension_set_id,
            authoring_package.extension_set,
        ),
        extension_materializations=extension_materializations,
        base_report=base_report,
        candidate=candidate,
    )


_ALLOWED_STAGE_KINDS: dict[ScientificStage, frozenset[CapabilityKind]] = {
    ScientificStage.PREPARE: frozenset({CapabilityKind.SOURCE, CapabilityKind.SIMULATOR}),
    ScientificStage.ACQUIRE: frozenset(
        {
            CapabilityKind.SOURCE,
            CapabilityKind.SIMULATOR,
            CapabilityKind.ACTUATOR,
            CapabilityKind.OBSERVATION_OPERATOR,
        }
    ),
    ScientificStage.TRANSFORM: frozenset({CapabilityKind.TRANSFORM}),
    ScientificStage.DEVELOP: frozenset(
        {
            CapabilityKind.ANALYSIS,
            CapabilityKind.PROSPECTIVE_NOMINATOR,
            # A prospective method may issue its frozen, development-visible
            # prediction roster here and adjudicate only in a later EVALUATE
            # task. Outcome-access and graph chronology remain independently
            # checked; this does not authorize target evidence in development.
            CapabilityKind.HYPOTHESIS_ADJUDICATOR,
            CapabilityKind.LAW_IDENTIFIER,
            CapabilityKind.RESPONSE_ALGEBRA_IDENTIFIER,
            CapabilityKind.CONTROLLER_SYNTHESIZER,
        }
    ),
    ScientificStage.FALSIFY: frozenset(
        {
            CapabilityKind.FALSIFIER,
            CapabilityKind.TRANSPORT_TESTER,
            CapabilityKind.DISCREPANCY_ESTIMATOR,
        }
    ),
    ScientificStage.QUALIFY: frozenset(
        {
            CapabilityKind.LAW_IDENTIFIER,
            CapabilityKind.NUMERICAL_QUALIFIER,
            CapabilityKind.OBSERVATION_OPERATOR,
            CapabilityKind.DISCREPANCY_ESTIMATOR,
        }
    ),
    ScientificStage.FREEZE: frozenset(
        {
            CapabilityKind.EVALUATOR,
            CapabilityKind.EXPERIMENT_DESIGNER,
            CapabilityKind.TRANSFORM,
            CapabilityKind.REPORTER,
        }
    ),
    ScientificStage.EVALUATE: frozenset(
        {CapabilityKind.EVALUATOR, CapabilityKind.HYPOTHESIS_ADJUDICATOR}
    ),
    ScientificStage.REVEAL: frozenset({CapabilityKind.EVALUATOR}),
    ScientificStage.SYNTHESIZE: frozenset(
        {
            CapabilityKind.ANALYSIS,
            CapabilityKind.ATLAS_ASSEMBLER,
            CapabilityKind.HYPOTHESIS_SYNTHESIZER,
            CapabilityKind.REPORTER,
        }
    ),
    ScientificStage.ADMISSION: frozenset(
        {CapabilityKind.ADMISSION_EVALUATOR, CapabilityKind.ONLINE_GATE_EVALUATOR}
    ),
    ScientificStage.CONTROLLER: frozenset(
        {CapabilityKind.CONTROLLER_SYNTHESIZER, CapabilityKind.ONLINE_GATE_EVALUATOR}
    ),
    ScientificStage.REPORT: frozenset({CapabilityKind.REPORTER}),
    ScientificStage.EXPLORE: frozenset(
        {
            CapabilityKind.ANALYSIS,
            CapabilityKind.ANOMALY_DETECTOR,
            CapabilityKind.PORTFOLIO_PLANNER,
        }
    ),
}

_REQUIRED_ISSUE_INPUTS = (
    "accountable-human-proposer-attestation",
    "clean-commit-or-exact-source-closure",
    "custody-publication-authority",
    "exact-materialization-revalidation",
)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def make_candidate_diagnostic(
    diagnostic_class: CandidateDiagnosticClass,
    code: CandidateDiagnosticCode,
    field_path: str,
    message: str,
) -> CandidateDiagnostic:
    """Build one stable diagnostic for pure or composed readiness checks."""

    identity = _digest(
        {
            "class": diagnostic_class.value,
            "code": code.value,
            "field_path": field_path,
            "message": message,
        }
    )
    return CandidateDiagnostic(
        diagnostic_id=f"diagnostic.{identity[:32]}",
        diagnostic_class=diagnostic_class,
        code=code,
        field_path=field_path,
        message=message,
    )


_diagnostic = make_candidate_diagnostic


def _deduplicate_diagnostics(
    diagnostics: list[CandidateDiagnostic],
) -> list[CandidateDiagnostic]:
    return list({value.diagnostic_id: value for value in diagnostics}.values())


def _required_obligation_ids(experiment: ExperimentSpec) -> tuple[str, ...]:
    obligations = experiment.obligations
    identifiers = {
        obligations.obligations_id,
        obligations.support.support_id,
        obligations.validity.validity_id,
        obligations.uncertainty.uncertainty_id,
        obligations.closure.closure_id,
        obligations.structural_convergence.convergence_id,
        obligations.computability.computability_id,
        experiment.reveal_barrier.barrier_id,
        *(value.falsifier_id for value in obligations.falsifiers),
        *(value.control_id for value in experiment.controls),
        *(value.goal_id for value in experiment.precision_goals),
        *(value.cutoff_id for value in experiment.information_cutoffs),
        *(
            f"{experiment.experiment_id}.{suffix}"
            for suffix in (
                "admission",
                "authority",
                "causal-no-outcome-construction",
                "conditional-nonattempt",
                "independent-split",
                "mandatory-hold",
                "reachability",
                "required-outputs",
                "seal-reveal-separation",
                "source-readiness",
                "stopping",
            )
        ),
    }
    if obligations.discrepancy is not None:
        identifiers.add(obligations.discrepancy.discrepancy_id)
    return tuple(sorted(identifiers))


def required_candidate_obligation_ids(
    experiment: ExperimentSpec,
    protocol: ProtocolTemplate,
) -> tuple[str, ...]:
    """Public deterministic compatibility map to canonical obligation owners."""

    return tuple(
        sorted(
            set(_required_obligation_ids(experiment))
            | {obligation_id for step in protocol.steps for obligation_id in step.obligation_ids}
        )
    )


def _design_diagnostics(
    draft: StudyDraft,
    context: CandidateCompilationContext,
) -> list[CandidateDiagnostic]:
    diagnostics: list[CandidateDiagnostic] = []
    decision_cutoffs = {value.information_cutoff.fingerprint() for value in draft.design_inputs}
    if len(decision_cutoffs) != 1:
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                "design_inputs.information_cutoff",
                (
                    "the current approval contract requires one exact shared "
                    "design-input information cutoff"
                ),
            )
        )
    known = {value.input_id: value for value in context.known_design_inputs}
    for value in draft.design_inputs:
        prior = known.get(value.input_id)
        if prior is not None and prior != value:
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.FATAL,
                    CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                    f"design_inputs.{value.input_id}",
                    "design input identity conflicts with registered lineage",
                )
            )
        known[value.input_id] = value
    declared = set(draft.design_origin.declared_input_ids)
    missing_declared = declared - set(known)
    for input_id in sorted(missing_declared):
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                f"design_origin.declared_input_ids.{input_id}",
                "declared design input is absent from the transitive ledger",
            )
        )
    closure: set[str] = set()
    state: dict[str, int] = {}
    for root_id in sorted(declared):
        pending: list[tuple[str, bool]] = [(root_id, False)]
        while pending:
            input_id, exiting = pending.pop()
            if exiting:
                state[input_id] = 2
                continue
            observed_state = state.get(input_id, 0)
            if observed_state == 2:
                continue
            if observed_state == 1:
                diagnostics.append(
                    _diagnostic(
                        CandidateDiagnosticClass.FATAL,
                        CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                        f"design_inputs.{input_id}.parent_input_ids",
                        "design-input lineage contains a cycle",
                    )
                )
                continue
            current_input = known.get(input_id)
            if current_input is None:
                continue
            state[input_id] = 1
            closure.add(input_id)
            pending.append((input_id, True))
            missing_parents = set(current_input.parent_input_ids) - set(known)
            for parent_id in sorted(missing_parents):
                diagnostics.append(
                    _diagnostic(
                        CandidateDiagnosticClass.FATAL,
                        CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                        f"design_inputs.{input_id}.parent_input_ids.{parent_id}",
                        "known transitive design lineage is incomplete",
                    )
                )
            pending.extend(
                (parent_id, False) for parent_id in reversed(current_input.parent_input_ids)
            )
    draft_ids = {value.input_id for value in draft.design_inputs}
    if closure and draft_ids != closure:
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                "design_inputs",
                "draft design-input ledger differs from its complete transitive closure",
            )
        )
    development_units = set(draft.development_unit_ids)
    evaluation_units = set(draft.evaluation_unit_ids)
    reused_units = development_units & evaluation_units
    for unit_id in sorted(reused_units):
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                f"evaluation_unit_ids.{unit_id}",
                "development-visible physical unit cannot enter evaluation",
            )
        )
    reused_seeds = set(draft.development_seed_ids) & set(draft.evaluation_seed_ids)
    for seed_id in sorted(reused_seeds):
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                f"evaluation_seed_ids.{seed_id}",
                "pilot or tuning seed cannot enter prospective evaluation",
            )
        )
    inherited_visibility = draft.design_origin.parent_visibility_ceiling
    for value in draft.design_inputs:
        inherited_visibility = VisibilityCeiling.most_restrictive(
            inherited_visibility,
            value.visibility_ceiling,
        )
        if (
            value.role is DesignInputRole.READINESS_METADATA
            and value.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.FATAL,
                    CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                    f"design_inputs.{value.input_id}.outcome_access",
                    "readiness metadata cannot carry value-level outcome access",
                )
            )
        if value.role is DesignInputRole.CLAIM_DERIVATION:
            if (
                value.outcome_access is not OutcomeAccess.EVALUATION_SEALED
                or value.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            ):
                diagnostics.append(
                    _diagnostic(
                        CandidateDiagnosticClass.FATAL,
                        CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                        f"design_inputs.{value.input_id}",
                        "claim-derivation inputs must remain sealed and prospective",
                    )
                )
        elif set(value.physical_unit_ids) & evaluation_units:
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.FATAL,
                    CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                    f"design_inputs.{value.input_id}.physical_unit_ids",
                    "inspected or design-visible unit cannot enter evaluation roster",
                )
            )
        if set(value.seed_ids) & set(draft.evaluation_seed_ids):
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.FATAL,
                    CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                    f"design_inputs.{value.input_id}.seed_ids",
                    "inspected or tuning seed cannot enter evaluation roster",
                )
            )
        if value.operator_id.startswith(("chatgpt", "codex", "openai")):
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.FATAL,
                    CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                    f"design_inputs.{value.input_id}.operator_id",
                    "advisory AI cannot be a design-input operator or provenance dependency",
                )
            )
    if (
        draft.design_origin.kind.value == "PROSPECTIVE_NOMINATION"
        and not draft.design_origin.requests_fresh_child
    ):
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                "design_origin.requests_fresh_child",
                "nomination-derived origin remains nonpromotable without a fresh child",
            )
        )
    if draft.conditional_successor is not None:
        conditional_units = set(draft.conditional_successor.evaluation_unit_ids)
        if conditional_units & development_units:
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.FATAL,
                    CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                    "conditional_successor.evaluation_unit_ids",
                    "conditional child reuses a development-visible unit",
                )
            )
    if draft.experiment is not None and not (
        draft.experiment.design_visibility_ceiling.is_at_least_as_restrictive_as(
            inherited_visibility
        )
    ):
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                "experiment.design_visibility_ceiling",
                "experiment lowers inherited design-input visibility",
            )
        )
    return diagnostics


def _component_diagnostics(draft: StudyDraft) -> list[CandidateDiagnostic]:
    diagnostics: list[CandidateDiagnostic] = []
    for decision in draft.unresolved_decisions:
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.SCIENTIFIC_SPEC_INCOMPLETE,
                decision.field_path,
                f"material decision remains unresolved: {decision.decision_id}",
            )
        )
    for name, value in (
        ("system", draft.system),
        ("experiment", draft.experiment),
        ("campaign", draft.campaign),
    ):
        if value is None:
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.FATAL,
                    CandidateDiagnosticCode.SCIENTIFIC_SPEC_INCOMPLETE,
                    name,
                    f"draft lacks its canonical {name} component",
                )
            )
    if draft.system is None or draft.experiment is None or draft.campaign is None:
        return diagnostics
    try:
        validate_experiment_against_system(draft.experiment, draft.system)
        if (
            draft.experiment.authorization_record_id is not None
            or draft.experiment.readiness is not ReadinessStatus.AUTHORITY_REQUIRED
        ):
            raise ValueError("candidate experiment must remain authority pending")
        if draft.system.system_id not in draft.campaign.system_ids:
            raise ValueError("campaign omits the candidate system")
        if draft.experiment.world_id not in draft.campaign.world_ids:
            raise ValueError("campaign omits the candidate evidence world")
        claim_ids = {claim.claim_id for claim in draft.experiment.claims}
        if not claim_ids.issubset(draft.campaign.target_claim_ids):
            raise ValueError("campaign omits candidate experiment claims")
        experiment_identity = ObjectIdentity.from_record(
            draft.experiment.experiment_id,
            draft.experiment,
        )
        matching_nodes = [
            node for node in draft.campaign.nodes if node.object_identity == experiment_identity
        ]
        if not matching_nodes:
            raise ValueError("campaign graph does not bind the candidate experiment identity")
    except ValueError as error:
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.SCIENTIFIC_SPEC_INCOMPLETE,
                "canonical_components",
                str(error),
            )
        )
    return diagnostics


def _capability_diagnostics(
    draft: StudyDraft,
    context: CandidateCompilationContext,
    template: StudyTemplate | None,
) -> list[CandidateDiagnostic]:
    if template is None:
        return [
            _diagnostic(
                CandidateDiagnosticClass.UNRESOLVED_READINESS,
                CandidateDiagnosticCode.CAPABILITY_REQUIRED,
                "dag_template_key",
                f"static programme template is unavailable: {draft.dag_template_key}",
            )
        ]
    diagnostics: list[CandidateDiagnostic] = []
    active_templates = [template]
    if draft.conditional_successor is not None:
        conditional_template = context.template(draft.conditional_successor.template_key)
        if conditional_template is not None:
            active_templates.append(conditional_template)
    selections = {value.selection_id: value for value in draft.capability_selections}
    required_ids = {
        f"{step.capability_key}@{step.capability_version}"
        for active_template in active_templates
        for step in active_template.protocol.steps
    }
    for selection_id in sorted(set(selections) - required_ids):
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.CAPABILITY_CONTRACT_MISMATCH,
                f"capability_selections.{selection_id}",
                "draft selects a capability outside the bounded programme template",
            )
        )
    step_entries: list[tuple[str, ProtocolStepTemplate, CandidateGraphNode]] = []
    for active_template in active_templates:
        nodes = {node.node_id: node for node in active_template.graph.nodes}
        for step in active_template.protocol.steps:
            selection_id = f"{step.capability_key}@{step.capability_version}"
            step_entries.append((selection_id, step, nodes[step.step_id]))
    for selection_id, step, graph_node in sorted(
        step_entries,
        key=lambda value: (value[0], value[1].step_id),
    ):
        selection = selections.get(selection_id)
        if selection is None:
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.UNRESOLVED_READINESS,
                    CandidateDiagnosticCode.CAPABILITY_REQUIRED,
                    f"capability_selections.{selection_id}",
                    "programme template capability has no static draft selection",
                )
            )
            continue
        try:
            manifest = context.registry.resolve(
                step.capability_key,
                step.capability_version,
            )
        except KeyError:
            required_input_edges = tuple(
                sorted(
                    (
                        edge.consumer_input_id,
                        edge.scientific_role.value,
                        edge.payload_schema,
                    )
                    for active_template in active_templates
                    for edge in active_template.graph.edges
                    if edge.consumer_node_id == step.step_id
                )
            )
            input_ports = ",".join(
                f"{port}:{role}:{schema}" for port, role, schema in required_input_edges
            )
            output_ports = ",".join(
                f"{value.output_id}:{value.payload_schema}"
                for value in sorted(step.outputs, key=lambda value: value.output_id)
            )
            required_visibilities = tuple(
                sorted(
                    {
                        edge.visibility_ceiling.value
                        for active_template in active_templates
                        for edge in active_template.graph.edges
                        if edge.consumer_node_id == step.step_id
                    }
                )
            )
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.UNRESOLVED_READINESS,
                    CandidateDiagnosticCode.CAPABILITY_REQUIRED,
                    f"capability_selections.{selection_id}",
                    (
                        "selected capability is not in the injected static registry; "
                        f"stage={step.stage.value}; "
                        f"input_ports={input_ports or 'none'}; "
                        f"output_ports={output_ports}; "
                        "permissions="
                        f"{','.join(sorted(value.value for value in step.required_permissions)) or 'none'}; "
                        "visibility="
                        f"{','.join(required_visibilities) or step.visibility_ceiling.value}; "
                        f"outcome_access={step.requested_outcome_access.value}; "
                        "conformance=static-fingerprinted-manifest,"
                        "protocol-contract,focused-adapter-suite"
                    ),
                )
            )
            continue
        reasons: list[str] = []
        if selection.implementation_sha256 != manifest.implementation_sha256:
            reasons.append("implementation digest differs")
        if manifest.kind not in _ALLOWED_STAGE_KINDS[step.stage]:
            reasons.append("capability kind is forbidden for the scientific stage")
        if (
            manifest.config_schema != step.config.config_schema
            or manifest.config_schema_sha256 != step.config.config_schema_sha256
        ):
            reasons.append("capability config schema differs")
        if not {output.payload_schema for output in step.outputs}.issubset(
            manifest.output_schema_ids
        ):
            reasons.append("protocol output schema is unsupported")
        input_schemas = {
            edge.payload_schema
            for active_template in active_templates
            for edge in active_template.graph.edges
            if edge.consumer_node_id == step.step_id
        }
        if not input_schemas.issubset(manifest.input_schema_ids):
            reasons.append("exact scientific input schema is unsupported")
        if not set(step.required_permissions).issubset(manifest.permissions):
            reasons.append("protocol permissions exceed the capability manifest")
        outcome_rank = {
            OutcomeAccess.OUTCOME_BLIND: 0,
            OutcomeAccess.EVALUATION_SEALED: 0,
            OutcomeAccess.DEVELOPMENT_VISIBLE: 1,
            OutcomeAccess.EVALUATOR_REVEAL: 2,
            OutcomeAccess.EVALUATION_REVEALED: 2,
            OutcomeAccess.PRIVILEGED_TRUTH: 3,
        }
        if (
            outcome_rank[step.requested_outcome_access]
            > outcome_rank[manifest.maximum_outcome_access]
        ):
            reasons.append("protocol outcome access exceeds the capability manifest")
        if not manifest.resource_ceiling.contains(step.resource_budget):
            reasons.append("protocol resources exceed the capability ceiling")
        if manifest.implementation_sha256 != graph_node.implementation_sha256:
            reasons.append("graph implementation digest differs")
        if reasons:
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.FATAL,
                    CandidateDiagnosticCode.CAPABILITY_CONTRACT_MISMATCH,
                    f"capability_selections.{selection_id}",
                    "; ".join(reasons),
                )
            )
    return diagnostics


def _source_diagnostics(
    draft: StudyDraft,
    context: CandidateCompilationContext,
    template: StudyTemplate | None,
) -> tuple[list[CandidateDiagnostic], tuple[ObjectIdentity, ...]]:
    diagnostics: list[CandidateDiagnostic] = []
    qualifications = {
        ObjectIdentity.from_record(value.receipt_id, value): value
        for value in context.qualifications
    }
    external_inputs = (
        {}
        if template is None
        else {value.input_id: value for value in template.graph.external_inputs}
    )
    template_edges = () if template is None else template.graph.edges
    role_map = {
        SourceMaterializationRole.PREPARED_MEDIUM: ScientificInputRole.PREPARED_MEDIUM,
        SourceMaterializationRole.OBSERVATION_STREAM: ScientificInputRole.SOURCE,
        SourceMaterializationRole.CALIBRATION: ScientificInputRole.QUALIFICATION,
        SourceMaterializationRole.RECEIVER_DEFINITION: ScientificInputRole.RECEIVER,
        SourceMaterializationRole.NUMERICAL_CONFIGURATION: ScientificInputRole.MODEL,
    }
    source_like_roles = frozenset(role_map.values())
    if not draft.source_materializations and not external_inputs:
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.UNRESOLVED_READINESS,
                CandidateDiagnosticCode.SOURCE_UNRESOLVED,
                "source_materializations",
                "draft does not bind an exact source materialization",
            )
        )
    declared_source_ids = {value.source_id for value in draft.source_materializations}
    for input_id, external_input in sorted(external_inputs.items()):
        if (
            external_input.scientific_role in source_like_roles
            and input_id not in declared_source_ids
        ):
            consumer_ports = ",".join(
                f"{edge.consumer_node_id}:{edge.consumer_input_id}"
                for edge in sorted(
                    (value for value in template_edges if value.external_input_id == input_id),
                    key=lambda value: value.edge_id,
                )
            )
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.UNRESOLVED_READINESS,
                    CandidateDiagnosticCode.SOURCE_UNRESOLVED,
                    f"source_materializations.{input_id}",
                    (
                        "scientific graph source input has no exact materialization lock; "
                        f"role={external_input.scientific_role.value}; "
                        f"consumer_ports={consumer_ports or 'none'}; "
                        f"schema={external_input.payload_schema}; "
                        f"visibility={external_input.visibility_ceiling.value}; "
                        f"outcome_access={external_input.outcome_access.value}; "
                        "conformance=exact-content,experiment-qualification,"
                        "bounded-read"
                    ),
                )
            )
    accepted: list[ObjectIdentity] = []
    for source in draft.source_materializations:
        field = f"source_materializations.{source.source_id}"
        bound_external_input = external_inputs.get(source.source_id)
        if bound_external_input is None:
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.FATAL,
                    CandidateDiagnosticCode.CAPABILITY_CONTRACT_MISMATCH,
                    field,
                    "source materialization is not consumed by the exact scientific graph",
                )
            )
        elif bound_external_input.scientific_role is not role_map[source.role]:
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.FATAL,
                    CandidateDiagnosticCode.CAPABILITY_CONTRACT_MISMATCH,
                    f"{field}.role",
                    "source materialization role differs from its exact graph input role",
                )
            )
        if (
            bound_external_input is not None
            and bound_external_input.outcome_access is OutcomeAccess.EVALUATION_SEALED
            and (
                bound_external_input.scientific_role is not ScientificInputRole.MODEL
                or bound_external_input.content_identity_policy is not ContentIdentityPolicy.EXACT_SHA256
                or bound_external_input.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            )
        ):
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.FATAL,
                    CandidateDiagnosticCode.CAPABILITY_CONTRACT_MISMATCH,
                    f"{field}.sealed-model",
                    "sealed cross-campaign input requires an exact prospective model identity",
                )
            )
        if source.access_disposition is SourceAccessDisposition.ACQUISITION_AUTHORITY_REQUIRED:
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.UNRESOLVED_READINESS,
                    CandidateDiagnosticCode.SOURCE_ACQUISITION_AUTHORITY_REQUIRED,
                    f"{field}.access_disposition",
                    "source bytes require separate acquisition authority",
                )
            )
            continue
        if source.access_disposition is SourceAccessDisposition.UNAVAILABLE:
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.UNRESOLVED_READINESS,
                    CandidateDiagnosticCode.SOURCE_UNRESOLVED,
                    f"{field}.access_disposition",
                    "source materialization is unavailable",
                )
            )
            continue
        identity = source.qualification_receipt
        qualification = None if identity is None else qualifications.get(identity)
        if qualification is None:
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.UNRESOLVED_READINESS,
                    CandidateDiagnosticCode.SOURCE_QUALIFICATION_REQUIRED,
                    f"{field}.qualification_receipt",
                    "source qualification receipt is absent or stale",
                )
            )
            continue
        assert identity is not None
        if (
            qualification.source_id != source.source_id
            or qualification.materialization != source.materialization
            or qualification.content_sha256 != source.content_sha256
            or qualification.evidence_world_id != source.evidence_world_id
            or qualification.observation_operator != source.observation_operator
            or qualification.numerical_view_ids != source.numerical_view_ids
            or (
                bound_external_input is not None
                and (
                    qualification.outcome_access is not bound_external_input.outcome_access
                    or qualification.visibility_ceiling
                    is not bound_external_input.visibility_ceiling
                )
            )
            or (
                draft.system is not None
                and (
                    qualification.evidence_world_id != draft.system.world.world_id
                    or (
                        bool(draft.system.numerical_views)
                        and qualification.numerical_view_ids
                        != tuple(value.view_id for value in draft.system.numerical_views)
                    )
                    or qualification.native_unit_ids
                    != tuple(sorted({value.native_unit for value in draft.system.quantities}))
                    or qualification.frame_ids
                    != tuple(sorted({value.coordinate_frame for value in draft.system.quantities}))
                    or qualification.clock_ids
                    != tuple(value.clock_id for value in draft.system.clocks)
                    or qualification.receiver_semantics_id != draft.system.relation.relation_id
                )
            )
            or (
                draft.experiment is not None
                and (
                    qualification.validity_contract_id
                    != draft.experiment.obligations.validity.validity_id
                    or qualification.uncertainty_contract_id
                    != draft.experiment.obligations.uncertainty.uncertainty_id
                )
            )
        ):
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.UNRESOLVED_READINESS,
                    CandidateDiagnosticCode.SOURCE_QUALIFICATION_REQUIRED,
                    f"{field}.qualification_receipt",
                    "source qualification is incompatible with the draft",
                )
            )
            continue
        accepted.append(identity)
    return diagnostics, tuple(sorted(accepted, key=lambda value: value.object_id))


def _resource_diagnostics(
    draft: StudyDraft,
    context: CandidateCompilationContext,
    template: StudyTemplate | None,
) -> list[CandidateDiagnostic]:
    if template is None:
        return []
    active_templates = [template]
    if draft.conditional_successor is not None:
        conditional_template = context.template(draft.conditional_successor.template_key)
        if conditional_template is not None:
            active_templates.append(conditional_template)
    requests = [
        step.resource_budget
        for active_template in active_templates
        for step in active_template.protocol.steps
    ]
    if draft.campaign is not None:
        requests.append(draft.campaign.budget)
    if all(draft.resource_ceiling.contains(value) for value in requests):
        return []
    return [
        _diagnostic(
            CandidateDiagnosticClass.FATAL,
            CandidateDiagnosticCode.RESOURCE_UNBOUNDED,
            "resource_ceiling",
            "campaign or protocol resource request exceeds the frozen draft ceiling",
        )
    ]


def _obligation_diagnostics(
    draft: StudyDraft,
    template: StudyTemplate | None,
) -> list[CandidateDiagnostic]:
    if template is None or draft.experiment is None:
        return []
    expected = set(
        required_candidate_obligation_ids(
            draft.experiment,
            template.protocol,
        )
    )
    observed = {binding.obligation_id for binding in template.coverage.bindings}
    missing = expected - observed
    extra = observed - expected
    diagnostics = [
        _diagnostic(
            CandidateDiagnosticClass.FATAL,
            CandidateDiagnosticCode.SCIENTIFIC_OBLIGATION_UNCOVERED,
            f"obligation_coverage.{obligation_id}",
            "canonical scientific obligation has no unique proof owner/output",
        )
        for obligation_id in sorted(missing)
    ]
    diagnostics.extend(
        _diagnostic(
            CandidateDiagnosticClass.FATAL,
            CandidateDiagnosticCode.SCIENTIFIC_OBLIGATION_UNCOVERED,
            f"obligation_coverage.{obligation_id}",
            "coverage contains an obligation outside the canonical compatibility map",
        )
        for obligation_id in sorted(extra)
    )
    return diagnostics


def _formal_gap_diagnostics(
    package: StudyDefinition,
    context: StandardCandidateCompilationContext,
) -> tuple[
    list[CandidateDiagnostic],
    FormalGapSourceCapabilityInventory | None,
]:
    """Validate derived applicability and exact candidate-graph lowering."""

    diagnostics: list[CandidateDiagnostic] = []
    draft = package.draft
    entry = package.entry_package
    if draft.system is None or draft.experiment is None:
        return diagnostics, None
    denominator_id = draft.system.system_id
    if entry.coverage.denominator_id != denominator_id:
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.FORMAL_GAP_BINDING_MISMATCH,
                "entry_package.coverage.denominator_id",
                "formal coverage denominator differs from the exact SystemSpec",
            )
        )
    inventory = context.source_inventory(denominator_id)
    if inventory is None:
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.UNRESOLVED_READINESS,
                CandidateDiagnosticCode.FORMAL_GAP_COVERAGE_REQUIRED,
                "formal_source_inventory",
                "standard candidate context lacks an exact denominator inventory",
            )
        )
        return diagnostics, None

    draft_sources = tuple(
        sorted(
            (value.materialization for value in draft.source_materializations),
            key=lambda value: value.object_id,
        )
    )
    if inventory.source_materializations != draft_sources:
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.FORMAL_GAP_BINDING_MISMATCH,
                "formal_source_inventory.source_materializations",
                "formal source inventory differs from the draft source locks",
            )
        )
    if inventory.evidence_world is not entry.checklist.portfolio_world:
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.FORMAL_GAP_BINDING_MISMATCH,
                "formal_source_inventory.evidence_world",
                "formal source inventory differs from the entry evidence world",
            )
        )
    if (
        EvidenceCeiling.lowest(
            inventory.requested_claim_ceiling,
            EvidenceCeiling.LOCAL_LAW,
        )
        is not inventory.requested_claim_ceiling
    ):
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.FORMAL_GAP_CEILING_EXCEEDED,
                "formal_source_inventory.requested_claim_ceiling",
                "standard formal methods are capped at LOCAL_LAW",
            )
        )

    derived = derive_formal_gap_applicability(entry.register, inventory)
    authored = {value.gap_id: value for value in entry.coverage.applicability}
    for value in derived:
        observed = authored.get(value.gap_id)
        if observed == value:
            continue
        code = CandidateDiagnosticCode.FORMAL_GAP_APPLICABILITY_MISMATCH
        message = "authored formal applicability differs from qualified source facts"
        if observed is not None and (
            set(value.present_operand_ids) - set(observed.present_operand_ids)
        ):
            code = CandidateDiagnosticCode.FORMAL_GAP_FEASIBLE_OPERAND_OMITTED
            message = "source-feasible formal operand was omitted from applicability"
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                code,
                f"entry_package.coverage.applicability.{value.gap_id}",
                message,
            )
        )

    known_units = set(draft.development_unit_ids) | set(draft.evaluation_unit_ids)
    unknown_units = set(inventory.independent_unit_ids) - known_units
    if unknown_units:
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.FORMAL_GAP_BINDING_MISMATCH,
                "formal_source_inventory.independent_unit_ids",
                f"formal inventory names units outside the draft: {sorted(unknown_units)}",
            )
        )
    known_views = {value.view_id for value in draft.system.numerical_views}
    unknown_views = set(inventory.numerical_view_ids) - known_views
    if unknown_views:
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.FORMAL_GAP_BINDING_MISMATCH,
                "formal_source_inventory.numerical_view_ids",
                f"formal inventory names views outside the system: {sorted(unknown_views)}",
            )
        )

    template = context.base.template(draft.dag_template_key)
    if template is None:
        return diagnostics, inventory
    nodes = {value.node_id: value for value in template.graph.nodes}
    steps = {value.step_id: value for value in template.protocol.steps}
    edges = {value.edge_id: value for value in template.graph.edges}
    coverage = {value.obligation_id: value for value in template.coverage.bindings}
    experiment_controls = {value.control_id for value in draft.experiment.controls}
    selections = {
        (value.capability_key, value.capability_version): value
        for value in draft.capability_selections
    }
    gaps = {value.gap_id: value for value in entry.register.gaps}
    for assignment in entry.coverage.assignments:
        if assignment.disposition is not FormalGapCoverageDisposition.TEST_IN_THIS_ACT:
            continue
        gap = gaps[assignment.gap_id]
        selected_controls = set(assignment.selected_control_ids)
        if not selected_controls.issubset(experiment_controls):
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.FATAL,
                    CandidateDiagnosticCode.FORMAL_GAP_BINDING_MISMATCH,
                    f"entry_package.coverage.assignments.{gap.gap_id}.selected_control_ids",
                    "formal controls are not exact ExperimentSpec controls",
                )
            )
        for role, family_id in (
            (
                FormalMethodRole.ESTIMATOR,
                assignment.selected_estimator_family_id,
            ),
            (
                FormalMethodRole.MULTIPLICITY,
                assignment.selected_multiplicity_family_id,
            ),
        ):
            assert family_id is not None
            binding = context.formal_methods.resolve(role, family_id)
            field = f"entry_package.coverage.assignments.{gap.gap_id}.{role.value.lower()}"
            if binding is None or gap.gap_id not in binding.supported_gap_ids:
                diagnostics.append(
                    _diagnostic(
                        CandidateDiagnosticClass.FATAL,
                        CandidateDiagnosticCode.FORMAL_GAP_METHOD_REQUIRED,
                        field,
                        "formal family lacks a compatible registered method binding",
                    )
                )
                continue
            try:
                manifest = context.base.registry.resolve(
                    binding.capability_key,
                    binding.capability_version,
                )
            except KeyError:
                manifest = None
            selection = selections.get((binding.capability_key, binding.capability_version))
            if (
                manifest is None
                or manifest.implementation_sha256 != binding.implementation_sha256
                or selection is None
                or selection.implementation_sha256 != binding.implementation_sha256
            ):
                diagnostics.append(
                    _diagnostic(
                        CandidateDiagnosticClass.FATAL,
                        CandidateDiagnosticCode.FORMAL_GAP_METHOD_REQUIRED,
                        field,
                        "formal method version/implementation is absent or stale",
                    )
                )
            if (
                EvidenceCeiling.lowest(
                    inventory.requested_claim_ceiling,
                    gap.maximum_claim_ceiling,
                    binding.maximum_claim_ceiling,
                )
                is not inventory.requested_claim_ceiling
            ):
                diagnostics.append(
                    _diagnostic(
                        CandidateDiagnosticClass.FATAL,
                        CandidateDiagnosticCode.FORMAL_GAP_CEILING_EXCEEDED,
                        field,
                        "requested formal claim ceiling exceeds method or gap ceiling",
                    )
                )

        owners = set(assignment.adjudication_owner_ids)
        outputs = set(assignment.output_ids)
        obligations = set(assignment.obligation_ids)
        if not owners.issubset(nodes):
            diagnostics.append(
                _diagnostic(
                    CandidateDiagnosticClass.FATAL,
                    CandidateDiagnosticCode.FORMAL_GAP_GRAPH_BINDING_INVALID,
                    f"entry_package.coverage.assignments.{gap.gap_id}.adjudication_owner_ids",
                    "formal adjudication owner is not a candidate graph node",
                )
            )
        for owner_id in owners & set(steps):
            owner_outputs = {value.output_id for value in steps[owner_id].outputs}
            if not (outputs & owner_outputs):
                diagnostics.append(
                    _diagnostic(
                        CandidateDiagnosticClass.FATAL,
                        CandidateDiagnosticCode.FORMAL_GAP_GRAPH_BINDING_INVALID,
                        f"entry_package.coverage.assignments.{gap.gap_id}.output_ids",
                        "formal output is not owned by the named graph node",
                    )
                )
        for obligation_id in obligations:
            obligation_binding = coverage.get(obligation_id)
            if (
                obligation_binding is None
                or obligation_binding.proof_owner_node_id not in owners
                or obligation_binding.required_output_id not in outputs
            ):
                diagnostics.append(
                    _diagnostic(
                        CandidateDiagnosticClass.FATAL,
                        CandidateDiagnosticCode.FORMAL_GAP_GRAPH_BINDING_INVALID,
                        f"entry_package.coverage.assignments.{gap.gap_id}.obligation_ids",
                        "formal obligation lacks the exact graph owner/output binding",
                    )
                )
                continue
            contributor_edges = tuple(
                edges.get(edge_id) for edge_id in obligation_binding.contributor_edge_ids
            )
            if any(value is None for value in contributor_edges) or any(
                value is not None
                and value.consumer_node_id != obligation_binding.proof_owner_node_id
                for value in contributor_edges
            ):
                diagnostics.append(
                    _diagnostic(
                        CandidateDiagnosticClass.FATAL,
                        CandidateDiagnosticCode.FORMAL_GAP_GRAPH_BINDING_INVALID,
                        (f"entry_package.coverage.assignments.{gap.gap_id}.contributor_edge_ids"),
                        "formal contributor edge has the wrong proof owner",
                    )
                )
    return diagnostics, inventory


def _bind_source_inputs(
    graph: CandidateScientificGraph,
    sources: tuple[SourceMaterializationRef, ...],
) -> CandidateScientificGraph:
    source_by_id = {value.source_id: value for value in sources}
    bound: list[CandidateGraphExternalInput] = []
    bound_edges = list(graph.edges)
    for value in graph.external_inputs:
        if value.scientific_role is ScientificInputRole.PARENT_RECEIPT:
            bound.append(value)
            continue
        source = source_by_id.get(value.input_id)
        if source is None:
            bound.append(value)
            continue
        bound.append(
            replace(
                value,
                logical_artifact_id=source.materialization.object_id,
                content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
                expected_content_sha256=source.content_sha256,
            )
        )
        bound_edges = [
            (
                replace(
                    edge,
                    logical_artifact_id=source.materialization.object_id,
                )
                if edge.external_input_id == value.input_id
                else edge
            )
            for edge in bound_edges
        ]
    return replace(
        graph,
        external_inputs=tuple(sorted(bound, key=lambda item: item.input_id)),
        edges=tuple(sorted(bound_edges, key=lambda item: item.edge_id)),
    )


def _conditional_child(
    request: ConditionalChildRequest | None,
    context: CandidateCompilationContext,
    sources: tuple[SourceMaterializationRef, ...],
) -> tuple[FrozenConditionalChild | None, list[CandidateDiagnostic]]:
    if request is None:
        return None, []
    template = context.template(request.template_key)
    if template is None:
        return (
            None,
            [
                _diagnostic(
                    CandidateDiagnosticClass.UNRESOLVED_READINESS,
                    CandidateDiagnosticCode.CAPABILITY_REQUIRED,
                    "conditional_successor.template_key",
                    "conditional-child template is unavailable",
                )
            ],
        )
    try:
        bound_graph = _bind_source_inputs(template.graph, sources)
        frozen_payload = {
            "request": request,
            "protocol": template.protocol,
            "scientific_graph": bound_graph,
            "obligation_coverage": template.coverage,
        }
        value = FrozenConditionalChild(
            request=request,
            protocol=template.protocol,
            scientific_graph=bound_graph,
            obligation_coverage=template.coverage,
            scientific_template_sha256=_digest(frozen_payload),
            allowed_instantiation_fields=(
                "authority_id",
                "issue_id",
                "parent_receipt",
                "run_id",
                "task_id",
            ),
        )
    except ValueError as error:
        return (
            None,
            [
                _diagnostic(
                    CandidateDiagnosticClass.FATAL,
                    CandidateDiagnosticCode.SCIENTIFIC_SPEC_INCOMPLETE,
                    "conditional_successor",
                    str(error),
                )
            ],
        )
    return value, []


def compile_draft_candidate(
    *,
    draft: StudyDraft,
    authoring_materialization: AuthoringMaterializationIdentity,
    context: CandidateCompilationContext,
    readiness_diagnostics: tuple[CandidateDiagnostic, ...] = (),
) -> CandidateCompilationReport:
    """Compile with no filesystem, worker, catalog, authority or outcome access."""

    record_type_programmecandidate: type[DraftStudyCandidate] = (
        RetrospectiveStudyCandidate
        if isinstance(draft, RetrospectiveStudyDraft)
        else DraftStudyCandidate
    )
    record_type_candidatecompilationreport: type[CandidateCompilationReport] = (
        RetrospectiveCandidateReport
        if isinstance(draft, RetrospectiveStudyDraft)
        else CandidateCompilationReport
    )
    semantic_sha256 = draft.fingerprint()
    context_sha256 = context.fingerprint()
    template = context.template(draft.dag_template_key)
    if not isinstance(readiness_diagnostics, tuple) or any(
        not isinstance(value, CandidateDiagnostic) for value in readiness_diagnostics
    ):
        raise TypeError("readiness_diagnostics must contain CandidateDiagnostic records")
    diagnostics = [
        *readiness_diagnostics,
        *_component_diagnostics(draft),
        *_design_diagnostics(draft, context),
        *_capability_diagnostics(draft, context, template),
        *_resource_diagnostics(draft, context, template),
        *_obligation_diagnostics(draft, template),
    ]
    source_diagnostics, qualification_identities = _source_diagnostics(
        draft,
        context,
        template,
    )
    diagnostics.extend(source_diagnostics)
    conditional, conditional_diagnostics = _conditional_child(
        draft.conditional_successor,
        context,
        draft.source_materializations,
    )
    diagnostics.extend(conditional_diagnostics)
    if (
        draft.conditional_successor is not None
        and template is not None
        and draft.conditional_successor.parent_node_id
        not in {value.node_id for value in template.graph.nodes}
    ):
        diagnostics.append(
            _diagnostic(
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.SCIENTIFIC_SPEC_INCOMPLETE,
                "conditional_successor.parent_node_id",
                "conditional child names a parent outside the upstream graph",
            )
        )
    diagnostics = _deduplicate_diagnostics(diagnostics)
    proposed_graph = (
        None
        if template is None
        else _bind_source_inputs(template.graph, draft.source_materializations)
    )
    provisional_graph = None if proposed_graph is None else proposed_graph.fingerprint()
    provisional_coverage = None if template is None else template.coverage.fingerprint()
    expected_authority_gates = tuple(
        sorted(
            code.value
            for code in (
                CandidateDiagnosticCode.CUSTODY_PUBLICATION_AUTHORITY_REQUIRED,
                CandidateDiagnosticCode.EXPERIMENT_EXECUTION_AUTHORITY_REQUIRED,
                CandidateDiagnosticCode.OUTCOME_REVEAL_AUTHORITY_REQUIRED,
                CandidateDiagnosticCode.SCIENTIFIC_APPROVAL_AUTHORITY_REQUIRED,
            )
        )
    )
    design_codes = {
        CandidateDiagnosticCode.SCIENTIFIC_SPEC_INCOMPLETE,
        CandidateDiagnosticCode.SCIENTIFIC_OBLIGATION_UNCOVERED,
        CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
        CandidateDiagnosticCode.FORMAL_GAP_COVERAGE_REQUIRED,
        CandidateDiagnosticCode.FORMAL_GAP_APPLICABILITY_MISMATCH,
        CandidateDiagnosticCode.FORMAL_GAP_METHOD_REQUIRED,
        CandidateDiagnosticCode.FORMAL_GAP_BINDING_MISMATCH,
        CandidateDiagnosticCode.FORMAL_GAP_FEASIBLE_OPERAND_OMITTED,
        CandidateDiagnosticCode.FORMAL_GAP_GRAPH_BINDING_INVALID,
        CandidateDiagnosticCode.FORMAL_GAP_CEILING_EXCEEDED,
        CandidateDiagnosticCode.FORMAL_GAP_PANEL_INCOMPLETE,
    }
    source_codes = {
        CandidateDiagnosticCode.SOURCE_UNRESOLVED,
        CandidateDiagnosticCode.SOURCE_QUALIFICATION_REQUIRED,
        CandidateDiagnosticCode.SOURCE_ACQUISITION_AUTHORITY_REQUIRED,
    }
    design_state = (
        CandidateAxisState.BLOCKED
        if any(value.code in design_codes for value in diagnostics)
        else CandidateAxisState.READY
    )
    source_state = (
        CandidateAxisState.BLOCKED
        if any(value.code in source_codes for value in diagnostics)
        else CandidateAxisState.READY
    )
    freeze_state = (
        CandidateAxisState.BLOCKED
        if any(
            value.diagnostic_class
            in {
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticClass.UNRESOLVED_READINESS,
            }
            for value in diagnostics
        )
        else CandidateAxisState.READY
    )
    system_sha256 = None if draft.system is None else draft.system.fingerprint()
    experiment_sha256 = None if draft.experiment is None else draft.experiment.fingerprint()
    campaign_sha256 = None if draft.campaign is None else draft.campaign.fingerprint()
    protocol_sha256 = None if template is None else template.protocol.fingerprint()
    capability_locks_sha256 = _digest(draft.capability_selections)
    source_locks_sha256 = _digest(draft.source_materializations)
    resource_ceiling_sha256 = draft.resource_ceiling.fingerprint()

    fatal = any(value.diagnostic_class is CandidateDiagnosticClass.FATAL for value in diagnostics)
    unresolved = any(
        value.diagnostic_class is CandidateDiagnosticClass.UNRESOLVED_READINESS
        for value in diagnostics
    )
    if fatal or unresolved:
        disposition = (
            CandidateCompilationDisposition.INVALID_DRAFT
            if fatal
            else CandidateCompilationDisposition.UNRESOLVED_READINESS
        )
        return record_type_candidatecompilationreport(
            draft_id=draft.draft_id,
            disposition=disposition,
            authoring_materialization=authoring_materialization,
            semantic_config_sha256=semantic_sha256,
            resolved_context_sha256=context_sha256,
            implementation_sha256=context.implementation_sha256,
            system_sha256=system_sha256,
            experiment_sha256=experiment_sha256,
            campaign_sha256=campaign_sha256,
            protocol_sha256=protocol_sha256,
            capability_locks_sha256=capability_locks_sha256,
            source_locks_sha256=source_locks_sha256,
            resource_ceiling_sha256=resource_ceiling_sha256,
            capability_lock_ids=tuple(value.selection_id for value in draft.capability_selections),
            source_lock_ids=tuple(value.source_id for value in draft.source_materializations),
            compatible_qualification_receipt_ids=tuple(
                value.object_id for value in qualification_identities
            ),
            resource_ceiling=draft.resource_ceiling,
            diagnostics=tuple(sorted(diagnostics, key=lambda value: value.diagnostic_id)),
            candidate=None,
            provisional_protocol=None if template is None else template.protocol,
            provisional_graph=proposed_graph,
            provisional_coverage=None if template is None else template.coverage,
            provisional_graph_sha256=provisional_graph,
            provisional_coverage_sha256=provisional_coverage,
            design_validity_state=design_state,
            source_readiness_state=source_state,
            freeze_readiness_state=freeze_state,
            authority_state=CandidateAxisState.EXPECTED_AUTHORITY_GATE,
            expected_authority_gates=expected_authority_gates,
            required_issue_inputs=_REQUIRED_ISSUE_INPUTS,
        )

    assert draft.system is not None
    assert draft.experiment is not None
    assert draft.campaign is not None
    assert template is not None
    assert proposed_graph is not None
    graph = proposed_graph
    authority_codes = (
        CandidateDiagnosticCode.CUSTODY_PUBLICATION_AUTHORITY_REQUIRED,
        CandidateDiagnosticCode.EXPERIMENT_EXECUTION_AUTHORITY_REQUIRED,
        CandidateDiagnosticCode.OUTCOME_REVEAL_AUTHORITY_REQUIRED,
        CandidateDiagnosticCode.SCIENTIFIC_APPROVAL_AUTHORITY_REQUIRED,
    )
    diagnostics.extend(
        _diagnostic(
            CandidateDiagnosticClass.AUTHORITY_GATE,
            code,
            "expected_authority_gates",
            "expected post-candidate authority is separate and not granted",
        )
        for code in authority_codes
    )
    diagnostics.append(
        _diagnostic(
            CandidateDiagnosticClass.SUCCESS,
            CandidateDiagnosticCode.READY_TO_ISSUE,
            "candidate",
            "complete deterministic candidate is ready for separate issue validation",
        )
    )
    candidate_payload = {
        "raw": authoring_materialization,
        "semantic": semantic_sha256,
        "context": context_sha256,
        "implementation": context.implementation_sha256,
        "system": draft.system,
        "experiment": draft.experiment,
        "campaign": draft.campaign,
        "protocol": template.protocol,
        "graph": graph,
        "coverage": template.coverage,
        "conditional": conditional,
    }
    candidate_digest = _digest(candidate_payload)
    candidate = record_type_programmecandidate(
        candidate_id=f"candidate.{candidate_digest[:32]}",
        disposition=CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING,
        authoring_materialization=authoring_materialization,
        semantic_config_sha256=semantic_sha256,
        resolved_context_sha256=context_sha256,
        implementation_sha256=context.implementation_sha256,
        system=draft.system,
        experiment=draft.experiment,
        campaign=draft.campaign,
        protocol=template.protocol,
        scientific_graph=graph,
        obligation_coverage=template.coverage,
        design_origin=draft.design_origin,
        design_input_ids=tuple(value.input_id for value in draft.design_inputs),
        capability_locks=draft.capability_selections,
        source_locks=draft.source_materializations,
        source_qualification_receipts=qualification_identities,
        resource_ceiling=draft.resource_ceiling,
        conditional_successor=conditional,
        design_validity_state=CandidateAxisState.READY,
        source_readiness_state=CandidateAxisState.READY,
        freeze_readiness_state=CandidateAxisState.READY,
        authority_state=CandidateAxisState.EXPECTED_AUTHORITY_GATE,
        expected_authority_gates=expected_authority_gates,
        required_issue_inputs=_REQUIRED_ISSUE_INPUTS,
    )
    return record_type_candidatecompilationreport(
        draft_id=draft.draft_id,
        disposition=CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING,
        authoring_materialization=authoring_materialization,
        semantic_config_sha256=semantic_sha256,
        resolved_context_sha256=context_sha256,
        implementation_sha256=context.implementation_sha256,
        system_sha256=system_sha256,
        experiment_sha256=experiment_sha256,
        campaign_sha256=campaign_sha256,
        protocol_sha256=protocol_sha256,
        capability_locks_sha256=capability_locks_sha256,
        source_locks_sha256=source_locks_sha256,
        resource_ceiling_sha256=resource_ceiling_sha256,
        capability_lock_ids=tuple(value.selection_id for value in draft.capability_selections),
        source_lock_ids=tuple(value.source_id for value in draft.source_materializations),
        compatible_qualification_receipt_ids=tuple(
            value.object_id for value in qualification_identities
        ),
        resource_ceiling=draft.resource_ceiling,
        diagnostics=tuple(sorted(diagnostics, key=lambda value: value.diagnostic_id)),
        candidate=candidate,
        provisional_protocol=template.protocol,
        provisional_graph=graph,
        provisional_coverage=template.coverage,
        provisional_graph_sha256=graph.fingerprint(),
        provisional_coverage_sha256=template.coverage.fingerprint(),
        design_validity_state=CandidateAxisState.READY,
        source_readiness_state=CandidateAxisState.READY,
        freeze_readiness_state=CandidateAxisState.READY,
        authority_state=CandidateAxisState.EXPECTED_AUTHORITY_GATE,
        expected_authority_gates=expected_authority_gates,
        required_issue_inputs=_REQUIRED_ISSUE_INPUTS,
    )


def compile_study_candidate(
    *,
    authoring_package: StudyDefinition,
    authoring_materialization: AuthoringMaterializationIdentity,
    context: StandardCandidateCompilationContext,
    readiness_diagnostics: tuple[CandidateDiagnostic, ...] = (),
) -> StudyCompilationReport:
    """Compile the mandatory standard root without filesystem or outcome access."""

    record_type_standardprogrammecandidate: type[StudyCandidate] = (
        RetrospectiveStandardCandidate
        if isinstance(authoring_package, RetrospectiveAuthoringBase)
        else StudyCandidate
    )
    record_type_standardcandidatecompilationreport: type[StudyCompilationReport] = (
        RetrospectiveStandardReport
        if isinstance(authoring_package, RetrospectiveAuthoringBase)
        else StudyCompilationReport
    )
    validate_experiment_entry(
        authoring_package.draft,
        authoring_package.entry_package,
        for_issue=False,
    )
    formal_diagnostics, inventory = _formal_gap_diagnostics(
        authoring_package,
        context,
    )
    base_report = compile_draft_candidate(
        draft=authoring_package.draft,
        authoring_materialization=authoring_materialization,
        context=context.base,
        readiness_diagnostics=(
            *readiness_diagnostics,
            *formal_diagnostics,
        ),
    )
    standard_candidate: StudyCandidate | None = None
    package_identity = ObjectIdentity.from_record(
        authoring_package.package_id,
        authoring_package,
    )
    entry = authoring_package.entry_package
    entry_identity = ObjectIdentity.from_record(entry.package_id, entry)
    if base_report.candidate is not None and inventory is not None:
        register_identity = ObjectIdentity.from_record(
            entry.register.register_id,
            entry.register,
        )
        coverage_identity = ObjectIdentity.from_record(
            entry.coverage.coverage_id,
            entry.coverage,
        )
        method_catalog_identity = ObjectIdentity.from_record(
            context.formal_methods.catalog_id,
            context.formal_methods,
        )
        source_inventory_identity = ObjectIdentity.from_record(
            inventory.inventory_id,
            inventory,
        )
        candidate_seed = {
            "base_candidate": ObjectIdentity.from_record(
                base_report.candidate.candidate_id,
                base_report.candidate,
            ),
            "authoring_package": package_identity,
            "entry_package": entry_identity,
            "formal_gap_register": register_identity,
            "formal_gap_coverage": coverage_identity,
            "formal_method_catalog": method_catalog_identity,
            "formal_source_inventory": source_inventory_identity,
            "standard_context_sha256": context.fingerprint(),
        }
        candidate_digest = _digest(candidate_seed)
        standard_candidate = record_type_standardprogrammecandidate(
            candidate_id=f"standard-candidate.{candidate_digest[:32]}",
            base_candidate=base_report.candidate,
            authoring_package=package_identity,
            entry_package=entry_identity,
            formal_gap_register=register_identity,
            formal_gap_coverage=coverage_identity,
            formal_method_catalog=method_catalog_identity,
            formal_source_inventory=source_inventory_identity,
            standard_context_sha256=context.fingerprint(),
        )
    report_seed = {
        "authoring_package": package_identity,
        "entry_package": entry_identity,
        "standard_context_sha256": context.fingerprint(),
        "base_report": base_report,
        "candidate": standard_candidate,
    }
    report_digest = _digest(report_seed)
    return record_type_standardcandidatecompilationreport(
        report_id=f"standard-report.{report_digest[:32]}",
        authoring_package=package_identity,
        entry_package=entry_identity,
        standard_context_sha256=context.fingerprint(),
        base_report=base_report,
        candidate=standard_candidate,
    )


__all__ = [
    "AuthoringMaterializationIdentity",
    "CandidateAxisState",
    "CandidateCompilationContext",
    "CandidateCompilationDisposition",
    "CandidateCompilationReport",
    "CandidateDiagnostic",
    "CandidateDiagnosticClass",
    "CandidateDiagnosticCode",
    "CandidateGraphEdge",
    "CandidateGraphExternalInput",
    "CandidateGraphNode",
    "CandidateScientificGraph",
    "ContentIdentityPolicy",
    'FrozenConditionalChild',
    "ObligationCoverage",
    "ObligationCoverageBinding",
    'DraftStudyCandidate',
    "StandardCandidateCompilationContext",
    'StudyCompilationReport',
    'ExecutableStudyCompilationReport',
    'StudyCandidate',
    'ExecutableStudyCandidate',
    'bind_standard_candidate_compilation_report',
    "bind_standard_candidate_extensions",
    'compile_study_candidate',
    'StudyTemplate',
    "ScientificInputRole",
    'compile_draft_candidate',
    "make_candidate_diagnostic",
    "required_candidate_obligation_ids",
]


@dataclass(frozen=True, slots=True)
class RetrospectiveStudyCandidate(DraftStudyCandidate):
    """Explicit historical compatibility; the predecessor schema is unchanged."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retrospective-study-candidate'

    experiment: RetrospectiveExperimentSpec
    design_origin: RetrospectiveDesignOrigin

    def __post_init__(self) -> None:
        if type(self.experiment) is not RetrospectiveExperimentSpec:
            raise ValueError("RetrospectiveStudyCandidate requires its exact experiment schema")
        if type(self.design_origin) is not RetrospectiveDesignOrigin:
            raise ValueError(
                "RetrospectiveStudyCandidate requires its exact design_origin schema"
            )
        DraftStudyCandidate.__post_init__(self)


@dataclass(frozen=True, slots=True)
class RetrospectiveStandardCandidate(StudyCandidate):
    """Explicit historical compatibility; the predecessor schema is unchanged."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retrospective-standard-candidate'

    base_candidate: RetrospectiveStudyCandidate

    def __post_init__(self) -> None:
        if type(self.base_candidate) is not RetrospectiveStudyCandidate:
            raise ValueError(
                "RetrospectiveStandardCandidate requires its exact base_candidate schema"
            )
        StudyCandidate.__post_init__(self)


@dataclass(frozen=True, slots=True)
class RetrospectiveExtensionCandidate(ExecutableStudyCandidate):
    """Explicit historical compatibility; the predecessor schema is unchanged."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retrospective-extension-candidate'

    base_candidate: RetrospectiveStandardCandidate
    authoring_package: RetrospectiveAuthoringPackage

    def __post_init__(self) -> None:
        if type(self.base_candidate) is not RetrospectiveStandardCandidate:
            raise ValueError(
                "RetrospectiveExtensionCandidate requires its exact base_candidate schema"
            )
        if type(self.authoring_package) is not RetrospectiveAuthoringPackage:
            raise ValueError(
                "RetrospectiveExtensionCandidate requires its exact authoring_package schema"
            )
        ExecutableStudyCandidate.__post_init__(self)


@dataclass(frozen=True, slots=True)
class RetrospectiveCandidateReport(CandidateCompilationReport):
    """Explicit historical compatibility; the predecessor schema is unchanged."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retrospective-candidate-report'

    candidate: RetrospectiveStudyCandidate | None

    def __post_init__(self) -> None:
        if (
            self.candidate is not None
            and type(self.candidate) is not RetrospectiveStudyCandidate
        ):
            raise ValueError("RetrospectiveCandidateReport requires its exact candidate schema")
        CandidateCompilationReport.__post_init__(self)


@dataclass(frozen=True, slots=True)
class RetrospectiveStandardReport(StudyCompilationReport):
    """Explicit historical compatibility; the predecessor schema is unchanged."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retrospective-standard-report'

    base_report: RetrospectiveCandidateReport
    candidate: RetrospectiveStandardCandidate | None

    def __post_init__(self) -> None:
        if type(self.base_report) is not RetrospectiveCandidateReport:
            raise ValueError("RetrospectiveStandardReport requires its exact base_report schema")
        if self.candidate is not None and type(self.candidate) is not RetrospectiveStandardCandidate:
            raise ValueError("RetrospectiveStandardReport requires its exact candidate schema")
        StudyCompilationReport.__post_init__(self)


@dataclass(frozen=True, slots=True)
class RetrospectiveExtensionReport(ExecutableStudyCompilationReport):
    """Explicit historical compatibility; the predecessor schema is unchanged."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retrospective-extension-report'

    base_report: RetrospectiveStandardReport
    candidate: RetrospectiveExtensionCandidate | None

    def __post_init__(self) -> None:
        if type(self.base_report) is not RetrospectiveStandardReport:
            raise ValueError("RetrospectiveExtensionReport requires its exact base_report schema")
        if (
            self.candidate is not None
            and type(self.candidate) is not RetrospectiveExtensionCandidate
        ):
            raise ValueError("RetrospectiveExtensionReport requires its exact candidate schema")
        ExecutableStudyCompilationReport.__post_init__(self)
