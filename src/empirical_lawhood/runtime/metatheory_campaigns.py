"""Deterministic lowering of the closed executable-metatheory campaign."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.metatheory_campaign import MetatheoryCampaignProfile, MetatheoryCampaignStageRole, MetatheoryStageApplicability
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)


class MetatheoryCampaignCompilationDisposition(StrEnum):
    COMPILED_AUTHORITY_PENDING = "COMPILED_AUTHORITY_PENDING"
    CAPABILITY_BINDING_REQUIRED = "CAPABILITY_BINDING_REQUIRED"


@dataclass(frozen=True, slots=True)
class MetatheoryCampaignConditionEdge(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/metatheory-campaign-condition-edge'

    edge_id: str
    upstream_role: MetatheoryCampaignStageRole
    downstream_role: MetatheoryCampaignStageRole
    condition_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.edge_id, field_name="edge_id")
        validate_stable_id(self.condition_id, field_name="condition_id")
        if self.upstream_role is self.downstream_role:
            raise ValueError("metatheory condition edge cannot self-loop")


@dataclass(frozen=True, slots=True)
class MetatheoryCampaignCompilation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/metatheory-campaign-compilation'

    compilation_id: str
    profile: ObjectIdentity
    stage_applicability: tuple[MetatheoryStageApplicability, ...]
    protocol: ProtocolTemplate
    capability_registry: ObjectIdentity
    issued_payload_roster: tuple[ObjectIdentity, ...]
    condition_graph: tuple[MetatheoryCampaignConditionEdge, ...]
    predicted_terminal_schemas: tuple[str, ...]
    disposition: MetatheoryCampaignCompilationDisposition
    reason_codes: tuple[str, ...]
    grants_authority: bool
    executed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.compilation_id, field_name="compilation_id")
        require_sorted_unique_ids(
            self.stage_applicability,
            attribute="role",
            field_name="stage_applicability",
        )
        if {value.role for value in self.stage_applicability} != set(MetatheoryCampaignStageRole):
            raise ValueError("campaign compilation lacks stage applicability")
        require_sorted_unique_ids(
            self.issued_payload_roster,
            attribute="object_id",
            field_name="issued_payload_roster",
        )
        require_sorted_unique_ids(
            self.condition_graph,
            attribute="edge_id",
            field_name="condition_graph",
        )
        require_sorted_unique_strings(
            self.predicted_terminal_schemas,
            field_name="predicted_terminal_schemas",
            allow_empty=False,
        )
        for schema in self.predicted_terminal_schemas:
            validate_schema(schema)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.grants_authority or self.executed:
            raise ValueError("metatheory compilation cannot grant authority or execute")


@dataclass(frozen=True, slots=True)
class MetatheoryStageOwnerContract(CanonicalRecord):
    """Exact nonexecuting owner contract for one applicable campaign role."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/metatheory-stage-owner-contract'

    contract_id: str
    role: MetatheoryCampaignStageRole
    capability_key: str
    capability_version: str
    config_schema: str
    input_schema_ids: tuple[str, ...]
    output_schema_id: str
    canonical_owner: ObjectIdentity
    supporting_owners: tuple[ObjectIdentity, ...]
    implementation_sha256: str
    permissions: tuple[CapabilityPermission, ...]
    resource_budget: ResourceBudget
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.contract_id, field_name="contract_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_schema(self.config_schema)
        require_sorted_unique_strings(
            self.input_schema_ids,
            field_name="input_schema_ids",
            allow_empty=False,
        )
        for schema in self.input_schema_ids:
            validate_schema(schema)
        validate_schema(self.output_schema_id)
        require_sorted_unique_ids(
            self.supporting_owners,
            attribute="object_id",
            field_name="supporting_owners",
        )
        if self.canonical_owner in set(self.supporting_owners):
            raise ValueError("canonical owner cannot also be a supporting owner")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if tuple(sorted(set(self.permissions), key=lambda value: value.value)) != self.permissions:
            raise ValueError("metatheory owner permissions must be sorted and unique")
        expected_visibility = (
            VisibilityCeiling.OUTCOME_VISIBLE
            if self.outcome_access is OutcomeAccess.EVALUATION_REVEALED
            else VisibilityCeiling.PROSPECTIVE
        )
        if self.visibility_ceiling is not expected_visibility:
            raise ValueError("metatheory owner visibility differs from outcome access")


class MetatheoryStageOwnerRunner(Protocol):
    @property
    def owner_contract(self) -> MetatheoryStageOwnerContract: ...


@dataclass(frozen=True, slots=True)
class MetatheoryCampaignProvider:
    """Non-scientific coordinator validating and resolving stage owners."""

    compilation: MetatheoryCampaignCompilation
    profile: MetatheoryCampaignProfile
    capability_registry: CapabilityRegistry
    owner_contracts: tuple[MetatheoryStageOwnerContract, ...]
    stage_runners: tuple[MetatheoryStageOwnerRunner, ...]

    def __post_init__(self) -> None:
        if self.compilation.profile != ObjectIdentity.from_record(
            self.profile.profile_id,
            self.profile,
        ):
            raise ValueError("METATHEORY_COORDINATOR_PROFILE_MISMATCH")
        if self.compilation.capability_registry != ObjectIdentity.from_record(
            self.capability_registry.registry_id,
            self.capability_registry,
        ):
            raise ValueError("METATHEORY_COORDINATOR_REGISTRY_MISMATCH")
        applicable = {
            value.role for value in self.compilation.stage_applicability if value.applicable
        }
        by_role = {value.role: value for value in self.owner_contracts}
        if len(by_role) != len(self.owner_contracts) or set(by_role) != applicable:
            raise ValueError("METATHEORY_COORDINATOR_OWNER_ROSTER_MISMATCH")
        runner_by_role = {value.owner_contract.role: value for value in self.stage_runners}
        if len(runner_by_role) != len(self.stage_runners) or set(runner_by_role) != applicable:
            raise ValueError("METATHEORY_COORDINATOR_RUNNER_ROSTER_MISMATCH")
        if any(
            runner_by_role[role].owner_contract != contract for role, contract in by_role.items()
        ):
            raise ValueError("METATHEORY_COORDINATOR_RUNNER_CONTRACT_MISMATCH")

        capability_bindings = {value.role: value for value in self.profile.capabilities}
        owner_bindings = {value.role: value for value in self.profile.owners}
        artifact_bindings = {value.role: value for value in self.profile.artifacts}
        for role, contract in by_role.items():
            capability = capability_bindings[role]
            artifact = artifact_bindings[role]
            manifest = self.capability_registry.resolve(
                contract.capability_key,
                contract.capability_version,
            )
            if (
                capability.selection.capability_key != contract.capability_key
                or capability.selection.capability_version != contract.capability_version
                or capability.selection.implementation_sha256 != contract.implementation_sha256
                or capability.config_schema != contract.config_schema
                or capability.resource_budget != contract.resource_budget
                or owner_bindings[role].owner != contract.canonical_owner
                or artifact.payload_schema != contract.output_schema_id
                or artifact.outcome_access is not contract.outcome_access
                or artifact.visibility_ceiling is not contract.visibility_ceiling
                or manifest.implementation_sha256 != contract.implementation_sha256
                or manifest.config_schema != contract.config_schema
                or manifest.resource_ceiling != contract.resource_budget
                or manifest.maximum_outcome_access is not contract.outcome_access
                or not set(contract.input_schema_ids).issubset(manifest.input_schema_ids)
                or contract.output_schema_id not in manifest.output_schema_ids
                or contract.permissions != manifest.permissions
            ):
                raise ValueError(f"METATHEORY_COORDINATOR_PARITY_MISMATCH:{role.value}")

    def resolved_runners(self) -> tuple[MetatheoryStageOwnerRunner, ...]:
        """Return exact runners in scientific role order without invoking them."""

        by_role = {value.owner_contract.role: value for value in self.stage_runners}
        return tuple(
            value for role in MetatheoryCampaignStageRole if (value := by_role.get(role))
        )


_STAGE_KIND = {
    MetatheoryCampaignStageRole.SOURCE_PIPELINE: ScientificStage.ACQUIRE,
    MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION: ScientificStage.QUALIFY,
    MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION: ScientificStage.DEVELOP,
    MetatheoryCampaignStageRole.COORDINATE_EVALUATION: ScientificStage.QUALIFY,
    MetatheoryCampaignStageRole.LAW_QUALIFICATION_OPTIONAL: ScientificStage.QUALIFY,
    MetatheoryCampaignStageRole.ATLAS_QUALIFICATION_OPTIONAL: ScientificStage.SYNTHESIZE,
    MetatheoryCampaignStageRole.DECISION_ASSURANCE_OPTIONAL: ScientificStage.ADMISSION,
    MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL: ScientificStage.FALSIFY,
    MetatheoryCampaignStageRole.PREDICTION_ISSUE: ScientificStage.FREEZE,
    MetatheoryCampaignStageRole.TARGET_ACQUISITION: ScientificStage.ACQUIRE,
    # The barrier is platform-owned. The extension evaluator runs only after
    # that barrier and publishes a revealed conformance summary; it never owns
    # sealed-read or reveal authority.
    MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION: ScientificStage.EVALUATE,
    MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT: ScientificStage.SYNTHESIZE,
    MetatheoryCampaignStageRole.REPORT: ScientificStage.REPORT,
}


def _applicable_roles(
    profile: MetatheoryCampaignProfile,
) -> set[MetatheoryCampaignStageRole]:
    roles = {
        MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT,
        MetatheoryCampaignStageRole.REPORT,
    }
    if profile.source_pipeline_profile is not None:
        roles.update(
            {
                MetatheoryCampaignStageRole.SOURCE_PIPELINE,
                MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION,
            }
        )
    if profile.coordinate_challenge_spec is not None:
        roles.update(
            {
                MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION,
                MetatheoryCampaignStageRole.COORDINATE_EVALUATION,
            }
        )
    if profile.law_qualification_parent is not None:
        roles.add(MetatheoryCampaignStageRole.LAW_QUALIFICATION_OPTIONAL)
    if profile.atlas_qualification_spec is not None:
        roles.add(MetatheoryCampaignStageRole.ATLAS_QUALIFICATION_OPTIONAL)
    if profile.decision_assurance_spec is not None:
        roles.add(MetatheoryCampaignStageRole.DECISION_ASSURANCE_OPTIONAL)
    if profile.property_survival_spec is not None:
        roles.add(MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL)
    if profile.prediction_package is not None:
        roles.update(
            {
                MetatheoryCampaignStageRole.PREDICTION_ISSUE,
                MetatheoryCampaignStageRole.TARGET_ACQUISITION,
                MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION,
            }
        )
    return roles


def compile_metatheory_campaign_profile(
    *,
    profile: MetatheoryCampaignProfile,
    capability_registry: CapabilityRegistry,
) -> MetatheoryCampaignCompilation:
    """Compile the fixed role order without issue, contact, reveal or writes."""

    applicable = _applicable_roles(profile)
    applicability = tuple(
        sorted(
            (
                MetatheoryStageApplicability(
                    role=role,
                    applicable=role in applicable,
                    reason_codes=() if role in applicable else ("STAGE_CONDITION_FALSE",),
                )
                for role in MetatheoryCampaignStageRole
            ),
            key=lambda value: value.role,
        )
    )
    capabilities = {value.role: value for value in profile.capabilities}
    artifacts = {value.role: value for value in profile.artifacts}
    steps = []
    edges = []
    previous_role: MetatheoryCampaignStageRole | None = None
    previous_step_id: str | None = None
    for role in MetatheoryCampaignStageRole:
        if role not in applicable:
            continue
        binding = capabilities[role]
        artifact = artifacts[role]
        try:
            manifest = capability_registry.resolve(
                binding.selection.capability_key,
                binding.selection.capability_version,
            )
        except KeyError as error:
            raise ValueError(f"METATHEORY_CAPABILITY_UNREGISTERED:{role.value}") from error
        if (
            manifest.implementation_sha256 != binding.selection.implementation_sha256
            or manifest.config_schema != binding.config_schema
            or manifest.config_schema_sha256 != binding.config_schema_sha256
            or artifact.payload_schema not in manifest.output_schema_ids
        ):
            raise ValueError(f"METATHEORY_CAPABILITY_BINDING_DRIFT:{role.value}")
        step_id = f"metatheory-{role.value.lower().replace('_', '-')}"
        outcome_access = artifact.outcome_access
        barrier = BarrierKind.NONE
        if role is MetatheoryCampaignStageRole.PREDICTION_ISSUE:
            barrier = BarrierKind.FREEZE
        elif role is MetatheoryCampaignStageRole.TARGET_ACQUISITION:
            barrier = BarrierKind.AUTHORITY
        elif role is MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION:
            barrier = BarrierKind.REVEAL
        permissions = tuple(
            sorted(
                value
                for value in manifest.permissions
                if value
                in {
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.READ_SEALED_OUTCOMES,
                    CapabilityPermission.READ_OUTCOME_VISIBLE,
                    CapabilityPermission.REVEAL_OUTCOMES,
                }
            )
        )
        steps.append(
            ProtocolStepTemplate(
                step_id=step_id,
                stage=_STAGE_KIND[role],
                capability_key=manifest.capability_key,
                capability_version=manifest.capability_version,
                config=CapabilityConfigRef(
                    config_id=binding.config_id,
                    config_schema=binding.config_schema,
                    config_schema_sha256=binding.config_schema_sha256,
                    content_sha256=binding.config_content_sha256,
                    artifact_id=binding.config_artifact_id,
                ),
                dependency_step_ids=() if previous_step_id is None else (previous_step_id,),
                outputs=(
                    OutputTemplate(
                        output_id=artifact.output_id,
                        payload_schema=artifact.payload_schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type=artifact.media_type,
                        filename_suffix=artifact.filename_suffix,
                    ),
                ),
                required_permissions=permissions,
                requested_outcome_access=outcome_access,
                visibility_ceiling=artifact.visibility_ceiling,
                resource_budget=binding.resource_budget,
                resource_lock_ids=binding.resource_lock_ids,
                barrier=barrier,
                maximum_attempts=binding.maximum_attempts,
                obligation_ids=(f"obligation.metatheory.{role.value.lower()}",),
            )
        )
        if previous_role is not None:
            edges.append(
                MetatheoryCampaignConditionEdge(
                    edge_id=(f"condition-edge.{previous_role.value.lower()}.{role.value.lower()}"),
                    upstream_role=previous_role,
                    downstream_role=role,
                    condition_id="condition.predecessor-terminal-or-stopped",
                )
            )
        previous_role = role
        previous_step_id = step_id

    protocol = ProtocolTemplate(
        template_id=f"protocol.{profile.profile_id}",
        template_version=profile.profile_version,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )
    payloads = tuple(
        sorted(
            (
                value
                for value in (
                    profile.source_pipeline_profile,
                    profile.source_qualification_spec,
                    profile.coordinate_challenge_spec,
                    profile.law_qualification_parent,
                    profile.atlas_qualification_spec,
                    profile.decision_assurance_spec,
                    profile.dependence_spec,
                    profile.property_survival_spec,
                    profile.prediction_package,
                    profile.adjudication_spec,
                    profile.obstruction_profile,
                )
                if value is not None
            ),
            key=lambda value: value.object_id,
        )
    )
    registry_identity = ObjectIdentity.from_record(
        capability_registry.registry_id,
        capability_registry,
    )
    digest = sha256(
        profile.canonical_bytes()
        + protocol.canonical_bytes()
        + capability_registry.canonical_bytes()
    ).hexdigest()
    return MetatheoryCampaignCompilation(
        compilation_id=f"metatheory-compilation.{digest[:32]}",
        profile=ObjectIdentity.from_record(profile.profile_id, profile),
        stage_applicability=applicability,
        protocol=protocol,
        capability_registry=registry_identity,
        issued_payload_roster=payloads,
        condition_graph=tuple(sorted(edges, key=lambda value: value.edge_id)),
        predicted_terminal_schemas=tuple(
            sorted({artifacts[role].payload_schema for role in applicable})
        ),
        disposition=MetatheoryCampaignCompilationDisposition.COMPILED_AUTHORITY_PENDING,
        reason_codes=(),
        grants_authority=False,
        executed=False,
    )


__all__ = [
    'MetatheoryCampaignProvider',
    'MetatheoryCampaignCompilationDisposition',
    'MetatheoryCampaignCompilation',
    'MetatheoryCampaignConditionEdge',
    'MetatheoryStageOwnerContract',
    'MetatheoryStageOwnerRunner',
    "compile_metatheory_campaign_profile",
]
