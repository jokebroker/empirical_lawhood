"Exact G0 upstream and experiment-consumer binding.\n\nThis module binds experiment-owned translators and scientific configuration\nschemas to the already released shared services.  It deliberately contains no\narchive reader, simulator, law finalizer, admission evaluator, compiler or scheduler.\n"

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.multi_world_study import PRIMARY_MORPHISM_PROPERTY_ID, MultiWorldChildRole
from empirical_lawhood.runtime.response_law_release import MultiWorldReadinessConsumerPin, ReleaseConformanceAttestation, ResponseLawConsolidationRelease, MultiWorldReadinessReleaseAttestation, MultiWorldReadinessRelease


_CHILD_NAMESPACES = {
    MultiWorldChildRole.FAIR_MAST_ARCHIVE: ('empirical_lawhood.adapters.physical.mast_archive_response_qualification'),
    MultiWorldChildRole.MAPPED_DIRECT_TORAX: (
        'empirical_lawhood.adapters.simulators.mast_torax_state_transport'
    ),
    MultiWorldChildRole.GENERATED_GYM_TORAX: ('empirical_lawhood.adapters.simulators.tokamak_prospective_control'),
}

_REQUIRED_SIMULATOR_CAPABILITIES = frozenset(
    {
        "response-law.atlas",
        "response-law.controller_compile",
        "response-law.controller_programme",
        "response-law.controller_tick",
        "response-law.observation",
        "response-law.candidate_family",
        "response-law.component_qualification",
        "response-law.law_qualification",
        "response-law.admission_receipts",
        "response-law.controller_cohort_adjudication",
        "response-law.controller_unit_evaluation",
        "response-law.sealed_reference",
        "certification.certified-admission-margin-corpus-binder",
        "certification.gate-margin",
        "controller.action-aware-nested-controller-use",
        "controller.prospective-binding",
        "qualification.action-preparation-recurrence",
        "qualification.recurrence-guarded-response-law-finalizer",
        "reference.commitment-membership",
        "reference.finite-chart-class",
        "runtime.deadline-free-resource-recovery",
    }
)

_REQUIRED_GENERATED_ONLY_CAPABILITIES = frozenset(
    {
        "acquisition.bounded-query-selector",
        "acquisition.farthest-point-continuation",
    }
)

_REQUIRED_ARCHIVE_CAPABILITIES = frozenset(
    {
        "response-law.observation",
        "response-law.candidate_family",
        "response-law.component_qualification",
        "response-law.law_qualification",
    }
)

_REQUIRED_JOINT_CAPABILITIES = frozenset(
    {
        "adjudication.nonpooling-joint",
        "bundle.multi-world-issue-recovery",
        "morphism.archive-overlap-controls",
    }
)

_REQUIRED_NODE_FAMILIES = (
    "archive-exposure-staging",
    "archive-late-reveal-evaluation",
    "archive-to-torax-mapping",
    "atlas-controller-admission-reachability",
    "common-and-adaptive-acquisition",
    "confirmatory-cells",
    "denominator-algebra-calibration",
    "final-arm-qualification",
    "joint-adjudication-report",
    "mapped-excluded-development",
    "programme-compile-binding-tick",
    "property-fan-in",
    "reference-cells-class",
    "resource-recovery",
    "roster-domain-freeze",
    "simulator-reveal-evaluation",
    "source-runtime-revalidation",
)


@dataclass(frozen=True, slots=True)
class MultiWorldStudyChildPackageBinding(CanonicalRecord):
    """One child translator/configuration boundary and its shared-service route."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/multi-world-response-study/contracts/multi-world-study-child-package-binding'

    child_id: str
    role: MultiWorldChildRole
    implementation_namespace: str
    independent_unit_schema: str
    roster_schema: str
    scientific_config_schemas: tuple[str, ...]
    method_config_schemas: tuple[str, ...]
    outcome_domain_ids: tuple[str, ...]
    seal_schema_ids: tuple[str, ...]
    locally_owned_role_ids: tuple[str, ...]
    shared_capabilities: tuple[ObjectIdentity, ...]
    maximum_pre_reveal_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.child_id, field_name="child_id")
        if self.implementation_namespace != _CHILD_NAMESPACES[self.role]:
            raise ValueError("flagship child is bound to an unaccepted package namespace")
        for value in (
            self.independent_unit_schema,
            self.roster_schema,
            *self.scientific_config_schemas,
            *self.method_config_schemas,
            *self.seal_schema_ids,
        ):
            validate_schema(value)
        for name, values in (
            ("scientific_config_schemas", self.scientific_config_schemas),
            ("method_config_schemas", self.method_config_schemas),
            ("outcome_domain_ids", self.outcome_domain_ids),
            ("seal_schema_ids", self.seal_schema_ids),
            ("locally_owned_role_ids", self.locally_owned_role_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        require_sorted_unique_ids(
            self.shared_capabilities,
            attribute="object_id",
            field_name="shared_capabilities",
        )
        capability_ids = {value.object_id for value in self.shared_capabilities}
        if self.role is MultiWorldChildRole.FAIR_MAST_ARCHIVE:
            if not _REQUIRED_ARCHIVE_CAPABILITIES.issubset(capability_ids):
                raise ValueError("archive child omits the shared measurement through local law route")
            if capability_ids & {
                "response-law.controller_compile",
                "response-law.controller_tick",
                "response-law.controller_unit_evaluation",
            }:
                raise ValueError("archive child is incorrectly bound to a controller route")
        else:
            if not _REQUIRED_SIMULATOR_CAPABILITIES.issubset(capability_ids):
                raise ValueError("simulator child omits a shared law/controller/controller use service")
            if (
                self.role is MultiWorldChildRole.GENERATED_GYM_TORAX
                and not _REQUIRED_GENERATED_ONLY_CAPABILITIES.issubset(capability_ids)
            ):
                raise ValueError("generated child omits bounded acquisition services")
        if self.maximum_pre_reveal_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("flagship child pre-reveal access must remain evaluation-sealed")


@dataclass(frozen=True, slots=True)
class MultiWorldStudyGraphBinding(CanonicalRecord):
    """Frozen child isolation, causal cutoffs and reveal topology."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/multi-world-response-study/contracts/multi-world-study-graph-binding'

    graph_binding_id: str
    archive_child_id: str
    mapped_child_id: str
    generated_child_id: str
    node_family_ids: tuple[str, ...]
    simulator_reveal_domain_ids: tuple[str, ...]
    archive_reveal_domain_id: str
    forbidden_edge_ids: tuple[str, ...]
    primary_property_id: str
    archive_receiver_reveal_after_simulators: bool
    world_local_evidence_pooling_forbidden: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("graph_binding_id", self.graph_binding_id),
            ("archive_child_id", self.archive_child_id),
            ("mapped_child_id", self.mapped_child_id),
            ("generated_child_id", self.generated_child_id),
            ("archive_reveal_domain_id", self.archive_reveal_domain_id),
        ):
            validate_stable_id(value, field_name=name)
        if len({self.archive_child_id, self.mapped_child_id, self.generated_child_id}) != 3:
            raise ValueError("flagship graph collapses distinct child identities")
        require_sorted_unique_strings(
            self.node_family_ids,
            field_name="node_family_ids",
            allow_empty=False,
        )
        if self.node_family_ids != _REQUIRED_NODE_FAMILIES:
            raise ValueError("flagship graph node-family roster differs from the plan")
        require_sorted_unique_strings(
            self.simulator_reveal_domain_ids,
            field_name="simulator_reveal_domain_ids",
            allow_empty=False,
        )
        if len(self.simulator_reveal_domain_ids) != 2:
            raise ValueError("flagship graph requires two distinct simulator reveal domains")
        require_sorted_unique_strings(
            self.forbidden_edge_ids,
            field_name="forbidden_edge_ids",
            allow_empty=False,
        )
        required_forbidden = {
            "archive-outcome-to-mapping",
            "archive-outcome-to-simulator-commitment",
            "confirmatory-outcome-to-refit-or-recommit",
            "reference-outcome-to-fit-qualification-controller-admission-or-commitment",
            "simulator-outcome-to-archive-law-roster-or-threshold",
        }
        if (
            not required_forbidden.issubset(self.forbidden_edge_ids)
            or self.primary_property_id != PRIMARY_MORPHISM_PROPERTY_ID
            or not self.archive_receiver_reveal_after_simulators
            or not self.world_local_evidence_pooling_forbidden
        ):
            raise ValueError("flagship graph weakens a causal or non-pooling invariant")


@dataclass(frozen=True, slots=True)
class MultiWorldStudyConsumerPlatformBinding(CanonicalRecord):
    """Receipt-bound G0 consumer description over both released layers."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/multi-world-response-study/contracts/multi-world-study-consumer-platform-binding'

    binding_id: str
    consumer_plan: ObjectIdentity
    corrected_base_release: ObjectIdentity
    corrected_base_attestation: ObjectIdentity
    readiness_release: ObjectIdentity
    readiness_attestation: ObjectIdentity
    readiness_pin: ObjectIdentity
    corrected_base_source_sha256: str
    readiness_source_sha256: str
    shared_capabilities: tuple[ObjectIdentity, ...]
    readiness_proof_owner_bindings: tuple[ObjectIdentity, ...]
    readiness_semantic_output_contracts: tuple[ObjectIdentity, ...]
    children: tuple[MultiWorldStudyChildPackageBinding, ...]
    joint_package_namespace: str
    joint_shared_capabilities: tuple[ObjectIdentity, ...]
    graph: MultiWorldStudyGraphBinding
    archive_protection_schema: str
    barrier_plan_schema: str
    morphism_schema: str
    joint_adjudication_schema: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_sha256(
            self.corrected_base_source_sha256,
            field_name="corrected_base_source_sha256",
        )
        validate_sha256(self.readiness_source_sha256, field_name="readiness_source_sha256")
        for value in (
            self.archive_protection_schema,
            self.barrier_plan_schema,
            self.morphism_schema,
            self.joint_adjudication_schema,
        ):
            validate_schema(value)
        if self.joint_package_namespace != (
            'empirical_lawhood.adapters.composition.multi_world_response_study'
        ):
            raise ValueError("joint flagship package namespace differs")
        for name, values in (
            ("shared_capabilities", self.shared_capabilities),
            ("readiness_proof_owner_bindings", self.readiness_proof_owner_bindings),
            (
                "readiness_semantic_output_contracts",
                self.readiness_semantic_output_contracts,
            ),
            ("joint_shared_capabilities", self.joint_shared_capabilities),
        ):
            require_sorted_unique_ids(values, attribute="object_id", field_name=name)
            if not values:
                raise ValueError(f"{name} cannot be empty")
        require_sorted_unique_ids(self.children, attribute="child_id", field_name="children")
        if {value.role for value in self.children} != set(MultiWorldChildRole):
            raise ValueError("platform binding does not contain the exact three children")
        joint_ids = {value.object_id for value in self.joint_shared_capabilities}
        if not _REQUIRED_JOINT_CAPABILITIES.issubset(joint_ids):
            raise ValueError("joint parent omits released bundle/morphism/adjudication services")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("G0 consumer binding must be outcome-blind")


@dataclass(frozen=True, slots=True)
class MultiWorldStudyConsumerCompatibilityReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/multi-world-response-study/contracts/multi-world-study-consumer-compatibility-receipt'

    receipt_id: str
    binding: ObjectIdentity
    consumer_plan: ObjectIdentity
    corrected_base_release: ObjectIdentity
    readiness_release: ObjectIdentity
    readiness_pin: ObjectIdentity
    child_binding_ids: tuple[str, ...]
    shared_capability_count: int
    shared_service_fork_count: int
    archived_semantic_import_count: int
    target_local_decision_branch_count: int
    passed: bool
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_strings(
            self.child_binding_ids,
            field_name="child_binding_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.shared_capability_count <= 0:
            raise ValueError("compatibility receipt lacks shared capabilities")
        if (
            self.shared_service_fork_count
            or self.archived_semantic_import_count
            or self.target_local_decision_branch_count
        ):
            raise ValueError("compatibility receipt records a prohibited consumer fork")
        if not self.passed or self.reason_codes:
            raise ValueError("only a passing G0 receipt can be issued")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("G0 compatibility cannot access outcomes")


def verify_study_consumer_platform_binding(
    *,
    receipt_id: str,
    binding: MultiWorldStudyConsumerPlatformBinding,
    corrected_base_release: ResponseLawConsolidationRelease,
    corrected_base_attestation: ReleaseConformanceAttestation,
    readiness_release: MultiWorldReadinessRelease,
    readiness_attestation: MultiWorldReadinessReleaseAttestation,
    readiness_pin: MultiWorldReadinessConsumerPin,
) -> MultiWorldStudyConsumerCompatibilityReceipt:
    """Verify every upstream identity join and refuse a local shared-service fork."""

    base_id = ObjectIdentity.from_record(
        corrected_base_release.release_id,
        corrected_base_release,
    )
    base_attestation_id = ObjectIdentity.from_record(
        corrected_base_attestation.attestation_id,
        corrected_base_attestation,
    )
    readiness_id = ObjectIdentity.from_record(readiness_release.release_id, readiness_release)
    readiness_attestation_id = ObjectIdentity.from_record(
        readiness_attestation.attestation_id,
        readiness_attestation,
    )
    pin_id = ObjectIdentity.from_record(readiness_pin.pin_id, readiness_pin)
    expected_shared = tuple(
        sorted(
            corrected_base_release.public_capabilities + readiness_release.public_capabilities,
            key=lambda value: value.object_id,
        )
    )
    exact_joins = (
        binding.consumer_plan == readiness_pin.consumer_plan,
        binding.corrected_base_release == base_id,
        binding.corrected_base_attestation == base_attestation_id,
        binding.readiness_release == readiness_id,
        binding.readiness_attestation == readiness_attestation_id,
        binding.readiness_pin == pin_id,
        binding.corrected_base_source_sha256
        == corrected_base_release.source_closure.source_tree_sha256,
        binding.readiness_source_sha256 == readiness_release.source_closure.source_tree_sha256,
        binding.shared_capabilities == expected_shared,
        binding.readiness_proof_owner_bindings == readiness_release.proof_owner_bindings,
        binding.readiness_semantic_output_contracts == readiness_release.semantic_output_contracts,
        corrected_base_attestation.corrected_release == base_id,
        readiness_release.corrected_base_release == base_id,
        readiness_release.corrected_base_attestation == base_attestation_id,
        readiness_attestation.release == readiness_id,
        readiness_pin.corrected_base_release == base_id,
        readiness_pin.corrected_base_attestation == base_attestation_id,
        readiness_pin.readiness_release == readiness_id,
        readiness_pin.readiness_attestation == readiness_attestation_id,
    )
    if not all(exact_joins):
        raise ValueError("FLAGSHIP_READINESS_RELEASE_INCOMPATIBLE")

    released = {value.object_id: value for value in expected_shared}
    for child in binding.children:
        for capability in child.shared_capabilities:
            if released.get(capability.object_id) != capability:
                raise ValueError("FLAGSHIP_READINESS_RELEASE_INCOMPATIBLE")
    for capability in binding.joint_shared_capabilities:
        if released.get(capability.object_id) != capability:
            raise ValueError("FLAGSHIP_READINESS_RELEASE_INCOMPATIBLE")

    return MultiWorldStudyConsumerCompatibilityReceipt(
        receipt_id=receipt_id,
        binding=ObjectIdentity.from_record(binding.binding_id, binding),
        consumer_plan=binding.consumer_plan,
        corrected_base_release=base_id,
        readiness_release=readiness_id,
        readiness_pin=pin_id,
        child_binding_ids=tuple(value.child_id for value in binding.children),
        shared_capability_count=len(expected_shared),
        shared_service_fork_count=0,
        archived_semantic_import_count=0,
        target_local_decision_branch_count=0,
        passed=True,
        reason_codes=(),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


__all__ = [
    'MultiWorldStudyChildPackageBinding',
    'MultiWorldStudyGraphBinding',
    'MultiWorldStudyConsumerCompatibilityReceipt',
    'MultiWorldStudyConsumerPlatformBinding',
    'verify_study_consumer_platform_binding',
]
