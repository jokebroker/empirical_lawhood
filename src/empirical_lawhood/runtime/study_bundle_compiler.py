"""Pure compiler for a parent campaign, exact child candidates and joint fan-in.

This module is the bounded shared repair accepted by MAST--TORAX Phase 2.  It
does not issue, authorize, execute or reveal a programme.  Existing child
candidate and adjudication records remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.campaigns import CampaignSpec
from empirical_lawhood.planning.study_bundles import StudyBundleChild, StudyBundleSpec

from .capabilities import CapabilityKind, CapabilityRegistry
from .candidate_compiler import CandidateAxisState, ContentIdentityPolicy, DraftStudyCandidate, StudyTemplate, ScientificInputRole, StudyCandidate
from .plans import BarrierKind, ScientificStage


class StudyBundleCompilationDisposition(StrEnum):
    INVALID = "INVALID"
    COMPILED_AUTHORITY_PENDING = "COMPILED_AUTHORITY_PENDING"


@dataclass(frozen=True, slots=True)
class StudyBundleCompilationInput(CanonicalRecord):
    """Single strict public authoring root for deterministic bundle compilation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/study-bundle-compilation-input'

    package_id: str
    specification: StudyBundleSpec
    campaign: CampaignSpec
    child_candidates: tuple[StudyCandidate, ...]
    joint_template: StudyTemplate
    joint_registry: CapabilityRegistry

    def __post_init__(self) -> None:
        validate_stable_id(self.package_id, field_name="package_id")
        require_sorted_unique_ids(
            self.child_candidates,
            attribute="candidate_id",
            field_name="child_candidates",
        )
        if len(self.child_candidates) != len(self.specification.children):
            raise ValueError("bundle compilation input child count differs")


@dataclass(frozen=True, slots=True)
class CompiledStudyBundleChild(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-study-bundle-child'

    child_id: str
    campaign_node_id: str
    candidate: ObjectIdentity
    scientific_graph: ObjectIdentity
    result_producer_node_id: str
    result_output_id: str
    result_input_id: str
    result_schema: str

    def __post_init__(self) -> None:
        for name, value in (
            ("child_id", self.child_id),
            ("campaign_node_id", self.campaign_node_id),
            ("result_producer_node_id", self.result_producer_node_id),
            ("result_output_id", self.result_output_id),
            ("result_input_id", self.result_input_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class StudyBundleReceiptRequirement(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/study-bundle-receipt-requirement'

    requirement_id: str
    producer_node_id: str
    producer_output_id: str
    consumer_node_id: str
    consumer_input_id: str
    payload_schema: str

    def __post_init__(self) -> None:
        for name, value in (
            ("requirement_id", self.requirement_id),
            ("producer_node_id", self.producer_node_id),
            ("producer_output_id", self.producer_output_id),
            ("consumer_node_id", self.consumer_node_id),
            ("consumer_input_id", self.consumer_input_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class StudyBundleCandidate(CanonicalRecord):
    """Nonauthoritative exact bundle identity; every later authority stays open."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/study-bundle-candidate'

    candidate_id: str
    specification: ObjectIdentity
    campaign: ObjectIdentity
    children: tuple[CompiledStudyBundleChild, ...]
    joint_template: ObjectIdentity
    joint_registry: ObjectIdentity
    joint_graph_sha256: str
    aggregate_resource_ceiling: ResourceBudget
    receipt_requirements: tuple[StudyBundleReceiptRequirement, ...]
    authority_state: CandidateAxisState
    expected_authority_gate_ids: tuple[str, ...]
    issued: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        require_sorted_unique_ids(self.children, attribute="child_id", field_name="children")
        if len(self.children) < 2:
            raise ValueError("compiled bundle requires multiple world-local children")
        validate_sha256(self.joint_graph_sha256, field_name="joint_graph_sha256")
        require_sorted_unique_ids(
            self.receipt_requirements,
            attribute="requirement_id",
            field_name="receipt_requirements",
        )
        require_sorted_unique_strings(
            self.expected_authority_gate_ids,
            field_name="expected_authority_gate_ids",
            allow_empty=False,
        )
        if self.authority_state is not CandidateAxisState.EXPECTED_AUTHORITY_GATE:
            raise ValueError("programme bundle compilation cannot promote authority")
        if self.issued:
            raise ValueError("programme bundle candidate is not an issued programme")


@dataclass(frozen=True, slots=True)
class StudyBundleCompilationReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/study-bundle-compilation-report'

    report_id: str
    package: ObjectIdentity
    disposition: StudyBundleCompilationDisposition
    reason_codes: tuple[str, ...]
    candidate: StudyBundleCandidate | None

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        compiled = (
            self.disposition is StudyBundleCompilationDisposition.COMPILED_AUTHORITY_PENDING
        )
        if compiled != (self.candidate is not None):
            raise ValueError("bundle compilation disposition and candidate differ")
        if compiled and self.reason_codes:
            raise ValueError("compiled bundle cannot carry invalidity reasons")
        if not compiled and not self.reason_codes:
            raise ValueError("invalid bundle compilation requires reasons")


@dataclass(frozen=True, slots=True)
class StudyBundleReceiptBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/study-bundle-receipt-binding'

    requirement_id: str
    producer_node_id: str
    producer_output_id: str
    consumer_node_id: str
    consumer_input_id: str
    payload: ObjectIdentity
    receipt: ObjectIdentity

    def __post_init__(self) -> None:
        for name, value in (
            ("requirement_id", self.requirement_id),
            ("producer_node_id", self.producer_node_id),
            ("producer_output_id", self.producer_output_id),
            ("consumer_node_id", self.consumer_node_id),
            ("consumer_input_id", self.consumer_input_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class StudyBundleRecoveryVerification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/study-bundle-recovery-verification'

    verification_id: str
    expected_candidate: ObjectIdentity
    stored_candidate: ObjectIdentity
    graph_identity_preserved: bool
    scientific_identity_preserved: bool
    receipts_complete: bool
    recovery_allowed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.verification_id, field_name="verification_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected = (
            self.graph_identity_preserved
            and self.scientific_identity_preserved
            and self.receipts_complete
        )
        if self.recovery_allowed != expected:
            raise ValueError("bundle recovery decision differs from parity checks")
        if self.recovery_allowed == bool(self.reason_codes):
            raise ValueError("bundle recovery reasons differ from its decision")


def compile_study_bundle(
    package: StudyBundleCompilationInput,
) -> StudyBundleCompilationReport:
    """Validate and compile the exact fan-in without granting later authority."""

    spec = package.specification
    reasons: set[str] = set()
    _identity_matches(
        spec.campaign, package.campaign.campaign_id, package.campaign, reasons, "CAMPAIGN"
    )
    _identity_matches(
        spec.joint_descendant.template,
        package.joint_template.template_key,
        package.joint_template,
        reasons,
        "JOINT_TEMPLATE",
    )
    _identity_matches(
        spec.joint_descendant.registry,
        package.joint_registry.registry_id,
        package.joint_registry,
        reasons,
        "JOINT_REGISTRY",
    )

    candidates = {value.candidate_id: value for value in package.child_candidates}
    compiled_children: list[CompiledStudyBundleChild] = []
    for child in spec.children:
        standard = candidates.get(child.candidate.object_id)
        if standard is None:
            reasons.add("CHILD_CANDIDATE_MISSING_OR_FOREIGN")
            continue
        _identity_matches(
            child.candidate, standard.candidate_id, standard, reasons, "CHILD_CANDIDATE"
        )
        base = standard.base_candidate
        _validate_child(child, base, package.campaign, reasons)
        compiled_children.append(
            CompiledStudyBundleChild(
                child_id=child.child_id,
                campaign_node_id=child.campaign_node_id,
                candidate=ObjectIdentity.from_record(standard.candidate_id, standard),
                scientific_graph=ObjectIdentity.from_record(
                    base.scientific_graph.graph_id,
                    base.scientific_graph,
                ),
                result_producer_node_id=child.result_producer_node_id,
                result_output_id=child.result_output_id,
                result_input_id=child.result_input_id,
                result_schema=child.result_schema,
            )
        )
    _validate_joint_template(package, reasons)

    budgets = tuple(value.base_candidate.resource_ceiling for value in package.child_candidates) + (
        spec.joint_descendant.resource_ceiling,
    )
    aggregate = _aggregate_resource_budgets(budgets)
    if not package.campaign.budget.contains(aggregate):
        reasons.add("CAMPAIGN_RESOURCE_CEILING_EXCEEDED")

    package_identity = ObjectIdentity.from_record(package.package_id, package)
    if reasons:
        return StudyBundleCompilationReport(
            report_id=f"report.{package.package_id}",
            package=package_identity,
            disposition=StudyBundleCompilationDisposition.INVALID,
            reason_codes=tuple(sorted(reasons)),
            candidate=None,
        )

    children = tuple(sorted(compiled_children, key=lambda value: value.child_id))
    children_by_input = {value.result_input_id: value for value in children}
    receipts = tuple(
        sorted(
            (
                StudyBundleReceiptRequirement(
                    requirement_id=(
                        f"receipt.{children_by_input[edge.external_input_id].child_id}"
                        f".to-{edge.consumer_node_id}"
                    ),
                    producer_node_id=(
                        children_by_input[edge.external_input_id].result_producer_node_id
                    ),
                    producer_output_id=(children_by_input[edge.external_input_id].result_output_id),
                    consumer_node_id=edge.consumer_node_id,
                    consumer_input_id=edge.consumer_input_id,
                    payload_schema=edge.payload_schema,
                )
                for edge in package.joint_template.graph.edges
                if edge.external_input_id in children_by_input
            ),
            key=lambda value: value.requirement_id,
        )
    )
    candidate_id = f"bundle-candidate.{package.fingerprint()[:32]}"
    candidate = StudyBundleCandidate(
        candidate_id=candidate_id,
        specification=ObjectIdentity.from_record(spec.bundle_id, spec),
        campaign=ObjectIdentity.from_record(package.campaign.campaign_id, package.campaign),
        children=children,
        joint_template=ObjectIdentity.from_record(
            package.joint_template.template_key,
            package.joint_template,
        ),
        joint_registry=ObjectIdentity.from_record(
            package.joint_registry.registry_id,
            package.joint_registry,
        ),
        joint_graph_sha256=package.joint_template.graph.fingerprint(),
        aggregate_resource_ceiling=aggregate,
        receipt_requirements=receipts,
        authority_state=CandidateAxisState.EXPECTED_AUTHORITY_GATE,
        expected_authority_gate_ids=spec.joint_descendant.expected_authority_gate_ids,
        issued=False,
    )
    return StudyBundleCompilationReport(
        report_id=f"report.{package.package_id}",
        package=package_identity,
        disposition=StudyBundleCompilationDisposition.COMPILED_AUTHORITY_PENDING,
        reason_codes=(),
        candidate=candidate,
    )


def verify_study_bundle_recovery(
    *,
    expected: StudyBundleCandidate,
    stored: StudyBundleCandidate,
    receipts: tuple[StudyBundleReceiptBinding, ...],
) -> StudyBundleRecoveryVerification:
    """Prove exact graph/science replay and complete fan-in receipt resolution."""

    reasons: set[str] = set()
    if expected.joint_graph_sha256 != stored.joint_graph_sha256:
        reasons.add("STORED_GRAPH_DIFFERS_FROM_PREVIEW")
    if expected != stored:
        reasons.add("SCIENTIFIC_IDENTITY_CHANGED_DURING_RECOVERY")
    requirements = {value.requirement_id: value for value in expected.receipt_requirements}
    requirement_ids = set(requirements)
    observed_ids = [value.requirement_id for value in receipts]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != requirement_ids:
        reasons.add("TASK_RECEIPTS_INCOMPLETE_OR_DUPLICATED")
    for binding in receipts:
        requirement = requirements.get(binding.requirement_id)
        if requirement is None:
            continue
        if (
            binding.producer_node_id != requirement.producer_node_id
            or binding.producer_output_id != requirement.producer_output_id
            or binding.consumer_node_id != requirement.consumer_node_id
            or binding.consumer_input_id != requirement.consumer_input_id
            or binding.payload.object_schema != requirement.payload_schema
        ):
            reasons.add("TASK_RECEIPT_ROUTE_OR_SCHEMA_MISMATCH")
    return StudyBundleRecoveryVerification(
        verification_id=f"recovery.{expected.candidate_id}",
        expected_candidate=ObjectIdentity.from_record(expected.candidate_id, expected),
        stored_candidate=ObjectIdentity.from_record(stored.candidate_id, stored),
        graph_identity_preserved=expected.joint_graph_sha256 == stored.joint_graph_sha256,
        scientific_identity_preserved=expected == stored,
        receipts_complete=not {
            "TASK_RECEIPTS_INCOMPLETE_OR_DUPLICATED",
            "TASK_RECEIPT_ROUTE_OR_SCHEMA_MISMATCH",
        }
        & reasons,
        recovery_allowed=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


def _identity_matches(
    expected: ObjectIdentity,
    object_id: str,
    value: CanonicalRecord,
    reasons: set[str],
    prefix: str,
) -> None:
    if expected != ObjectIdentity.from_record(object_id, value):
        reasons.add(f"{prefix}_IDENTITY_DRIFT")


def _validate_child(
    child: StudyBundleChild,
    candidate: DraftStudyCandidate,
    campaign: CampaignSpec,
    reasons: set[str],
) -> None:
    if candidate.campaign != campaign:
        reasons.add("CHILD_CAMPAIGN_MISMATCH")
    if candidate.system.system_id != child.expected_system_id:
        reasons.add("CHILD_SYSTEM_SUBSTITUTION")
    if candidate.system.world.world_id != child.expected_world_id:
        reasons.add("CHILD_WORLD_SUBSTITUTION")
    if candidate.experiment.experiment_id != child.expected_experiment_id:
        reasons.add("CHILD_EXPERIMENT_SUBSTITUTION")
    if candidate.experiment.world_id != child.expected_world_id:
        reasons.add("CHILD_EXPERIMENT_WORLD_MISMATCH")
    try:
        node = campaign.node(child.campaign_node_id)
    except KeyError:
        reasons.add("CHILD_CAMPAIGN_NODE_MISSING")
        return
    expected_experiment = ObjectIdentity.from_record(
        candidate.experiment.experiment_id,
        candidate.experiment,
    )
    if node.object_identity != expected_experiment:
        reasons.add("CHILD_CAMPAIGN_NODE_IDENTITY_MISMATCH")
    try:
        producer = next(
            value
            for value in candidate.protocol.steps
            if value.step_id == child.result_producer_node_id
        )
    except StopIteration:
        reasons.add("CHILD_RESULT_PRODUCER_MISSING")
        return
    output = next(
        (value for value in producer.outputs if value.output_id == child.result_output_id),
        None,
    )
    if (
        output is None
        or output.payload_schema != child.result_schema
        or output.media_type != child.result_media_type
    ):
        reasons.add("CHILD_RESULT_OUTPUT_CONTRACT_MISMATCH")


def _validate_joint_template(
    package: StudyBundleCompilationInput,
    reasons: set[str],
) -> None:
    spec = package.specification
    template = package.joint_template
    nodes = {value.node_id: value for value in template.graph.nodes}
    steps = {value.step_id: value for value in template.protocol.steps}
    node = nodes.get(spec.joint_descendant.descendant_node_id)
    step = steps.get(spec.joint_descendant.descendant_node_id)
    if node is None or step is None:
        reasons.add("JOINT_DESCENDANT_NODE_MISMATCH")
        return
    expected_inputs = {value.result_input_id: value for value in spec.children}
    actual_inputs = {value.input_id: value for value in template.graph.external_inputs}
    if set(actual_inputs) != set(expected_inputs):
        reasons.add("JOINT_CHILD_INPUT_SET_MISMATCH")
    for input_id, child in expected_inputs.items():
        value = actual_inputs.get(input_id)
        if value is None:
            continue
        if (
            value.content_identity_policy is not ContentIdentityPolicy.PARENT_RECEIPT_SUBSTITUTION
            or value.scientific_role is not ScientificInputRole.PARENT_RECEIPT
            or value.payload_schema != child.result_schema
            or value.media_type != child.result_media_type
            or value.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or value.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            reasons.add("JOINT_CHILD_INPUT_CONTRACT_MISMATCH")
    external_edges = tuple(
        value for value in template.graph.edges if value.external_input_id is not None
    )
    if {value.external_input_id for value in external_edges} != set(expected_inputs) or any(
        edge.external_input_id not in expected_inputs
        or edge.scientific_role is not ScientificInputRole.PARENT_RECEIPT
        or edge.barrier is not BarrierKind.REVEAL
        for edge in external_edges
    ):
        reasons.add("JOINT_FAN_IN_EDGE_MISMATCH")
    if step.stage is not ScientificStage.REPORT:
        reasons.add("JOINT_STAGE_MISMATCH")
    for protocol_step in template.protocol.steps:
        graph_node = nodes[protocol_step.step_id]
        try:
            manifest = package.joint_registry.resolve(
                protocol_step.capability_key,
                protocol_step.capability_version,
            )
        except KeyError:
            reasons.add("JOINT_CAPABILITY_UNREGISTERED")
            continue
        expected_kind = (
            CapabilityKind.REPORTER
            if protocol_step.step_id == spec.joint_descendant.descendant_node_id
            else CapabilityKind.TRANSPORT_TESTER
        )
        input_schemas = {
            edge.payload_schema
            for edge in template.graph.edges
            if edge.consumer_node_id == protocol_step.step_id
        }
        if (
            manifest.kind is not expected_kind
            or manifest.implementation_sha256 != graph_node.implementation_sha256
            or manifest.config_schema != protocol_step.config.config_schema
            or manifest.config_schema_sha256 != protocol_step.config.config_schema_sha256
            or not input_schemas.issubset(manifest.input_schema_ids)
            or not {output.payload_schema for output in protocol_step.outputs}.issubset(
                manifest.output_schema_ids
            )
            or not manifest.resource_ceiling.contains(protocol_step.resource_budget)
        ):
            reasons.add("JOINT_CAPABILITY_CONTRACT_MISMATCH")
    joint_budget = _aggregate_resource_budgets(
        tuple(value.resource_budget for value in template.protocol.steps)
    )
    if joint_budget != spec.joint_descendant.resource_ceiling:
        reasons.add("JOINT_RESOURCE_CEILING_MISMATCH")
    child_nodes = tuple(value.campaign_node_id for value in spec.children)
    try:
        campaign_node = package.campaign.node(spec.joint_descendant.descendant_node_id)
    except KeyError:
        reasons.add("JOINT_CAMPAIGN_NODE_MISSING")
    else:
        if campaign_node.parent_node_ids != child_nodes:
            reasons.add("JOINT_CAMPAIGN_PARENT_MISMATCH")


def _aggregate_resource_budgets(budgets: tuple[ResourceBudget, ...]) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=max(value.cpu_cores for value in budgets),
        memory_bytes=max(value.memory_bytes for value in budgets),
        gpu_devices=max(value.gpu_devices for value in budgets),
        wall_time_seconds=sum(value.wall_time_seconds for value in budgets),
        source_scan_bytes=sum(value.source_scan_bytes for value in budgets),
        output_bytes=sum(value.output_bytes for value in budgets),
    )


__all__ = [
    'CompiledStudyBundleChild',
    'StudyBundleCandidate',
    'StudyBundleCompilationDisposition',
    'StudyBundleCompilationInput',
    'StudyBundleCompilationReport',
    'StudyBundleReceiptBinding',
    'StudyBundleReceiptRequirement',
    'StudyBundleRecoveryVerification',
    'compile_study_bundle',
    'verify_study_bundle_recovery',
]
