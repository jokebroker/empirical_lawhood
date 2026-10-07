"""Public, outcome-blind compilation for source and linked campaign profiles.

The services in this module join already-owned records.  They do not issue a
programme, open a source, construct authority, execute a task, reveal an
outcome, or reproduce any scientific finalizer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.dataset_authority import DatasetOperationRequest
from empirical_lawhood.planning.dataset_manifests import DatasetTransformationManifest
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection, EvidenceWorldProfileRegistry
from empirical_lawhood.planning.linked_campaign import LinkedCampaignPackageNode, LinkedCampaignPackageRole, LinkedCampaignProfile
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile

from .candidate_compiler import CandidateCompilationDisposition, DraftStudyCandidate, ExecutableStudyCompilationReport, ExecutableStudyCandidate
from .capabilities import CapabilityRegistry
from .datasets import DatasetCapabilityRegistry
from .evidence_profiles import EvidenceProfileRegistryResolver, ProfileFeasibilityDisposition, resolve_candidate_profile_feasibility
from .executable_bindings import ExecutableBindingRole, ExecutableCapabilityBinding, ExecutableCapabilityProviderFactoryRegistry, GeneratedExecutableBindingAggregate
from .linked_campaigns import LinkedCampaignStageEnvelope, build_linked_campaign_protocol
from .plans import ProtocolTemplate
from .source_pipelines import SourcePipelineCompilationDisposition, SourcePipelineCompilation, compile_source_pipeline
from .sources import SourceCapabilityManifest


class LinkedCampaignExecutableDisposition(StrEnum):
    COMPILED_AUTHORITY_PENDING = "COMPILED_AUTHORITY_PENDING"
    SOURCE_AUTHORITY_REQUIRED = "SOURCE_AUTHORITY_REQUIRED"
    EVIDENCE_PROFILE_BLOCKED = "EVIDENCE_PROFILE_BLOCKED"
    CANDIDATE_COMPILATION_BLOCKED = "CANDIDATE_COMPILATION_BLOCKED"
    EXECUTABLE_BINDING_REQUIRED = "EXECUTABLE_BINDING_REQUIRED"


class ExecutableBindingRequiredError(ValueError):
    """An installed profile compiler lacks an exact executable factory binding."""

    def __init__(self, missing_binding_ids: tuple[str, ...]) -> None:
        self.missing_binding_ids = tuple(sorted(set(missing_binding_ids)))
        if not self.missing_binding_ids:
            raise ValueError("executable-binding stop requires a missing binding")
        super().__init__("executable binding required: " + ",".join(self.missing_binding_ids))


_FACTORY_METHOD_BY_ROLE = {
    ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER: "build_provider",
    ExecutableBindingRole.PROFILE_COMPILER: "build_compiler",
    ExecutableBindingRole.LINKED_CAMPAIGN_COORDINATOR: "build_coordinator",
    ExecutableBindingRole.PROGRAMME_AUTHOR: 'build_study_author',
    ExecutableBindingRole.SOURCE_PROVIDER: "build_source_provider",
}


def _exact_factory_binding(
    factories: ExecutableCapabilityProviderFactoryRegistry,
    *,
    capability_key: str,
    capability_version: str,
    implementation_sha256: str,
    role: ExecutableBindingRole,
    required_output_schema_ids: frozenset[str],
    required_profile_schema_id: str | None = None,
    required_config_schema_id: str | None = None,
) -> ExecutableCapabilityBinding | None:
    """Resolve metadata and its concrete factory without constructing a provider."""

    binding = factories.aggregate.binding(capability_key, capability_version)
    if (
        binding is None
        or binding.capability_implementation_sha256 != implementation_sha256
        or binding.role is not role
        or not required_output_schema_ids.issubset(binding.output_schema_ids)
        or (
            required_profile_schema_id is not None
            and required_profile_schema_id
            not in {value.record_schema for value in binding.accepted_profile_types}
        )
        or (
            required_config_schema_id is not None
            and required_config_schema_id
            not in {value.record_schema for value in binding.accepted_config_types}
        )
    ):
        return None
    try:
        factory = factories.factory(binding.binding_id)
    except (KeyError, TypeError, ValueError):
        return None
    if getattr(factory, "binding", None) != binding or not callable(
        getattr(factory, _FACTORY_METHOD_BY_ROLE[role], None)
    ):
        return None
    return binding


@dataclass(frozen=True, slots=True)
class SourceProfileCompilerBinding(CanonicalRecord):
    """One installed, non-acquiring source/transform compiler context."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/source-profile-compiler-binding'

    binding_id: str
    source_manifest: SourceCapabilityManifest
    capability_registry: CapabilityRegistry
    dataset_registry: DatasetCapabilityRegistry

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        transform_ids = {value.registry_id for value in self.dataset_registry.transforms}
        capability_ids = {value.registry_id for value in self.capability_registry.capabilities}
        if not transform_ids or not transform_ids <= capability_ids:
            raise ValueError("source compiler dataset transforms lack generic manifests")

    def matches(self, profile: SourcePipelineProfile) -> bool:
        selection = profile.source_selection
        if (
            self.source_manifest.capability_key != selection.capability_key
            or self.source_manifest.capability_version != selection.capability_version
            or self.source_manifest.implementation_sha256 != selection.implementation_sha256
        ):
            return False
        transforms = {value.registry_id for value in self.dataset_registry.transforms}
        return all(
            value.dataset_transform_registry_id in transforms for value in profile.transform_edges
        )


@dataclass(frozen=True, slots=True)
class SourceProfileCompilerRegistry:
    """Closed installed compiler contexts; it contains no source adapter object."""

    bindings: tuple[SourceProfileCompilerBinding, ...]
    executable_factories: ExecutableCapabilityProviderFactoryRegistry

    def __post_init__(self) -> None:
        identifiers = tuple(value.binding_id for value in self.bindings)
        if not identifiers or tuple(sorted(set(identifiers))) != identifiers:
            raise ValueError("source compiler bindings must be sorted and unique")
        if not isinstance(
            self.executable_factories,
            ExecutableCapabilityProviderFactoryRegistry,
        ):
            raise TypeError("source compiler requires the authenticated factory registry")

    def _missing_executable_bindings(
        self,
        profile: SourcePipelineProfile,
    ) -> tuple[str, ...]:
        missing: list[str] = []
        source = profile.source_selection
        if (
            _exact_factory_binding(
                self.executable_factories,
                capability_key=source.capability_key,
                capability_version=source.capability_version,
                implementation_sha256=source.implementation_sha256,
                role=ExecutableBindingRole.SOURCE_PROVIDER,
                required_output_schema_ids=frozenset((profile.source_schema,)),
                required_profile_schema_id=SourcePipelineProfile.SCHEMA,
                required_config_schema_id=profile.source_config.object_schema,
            )
            is None
        ):
            missing.append(f"source-selection.{source.selection_id}")
        for edge in profile.transform_edges:
            selection = edge.selection
            runtime_binding = _exact_factory_binding(
                self.executable_factories,
                capability_key=selection.capability_key,
                capability_version=selection.capability_version,
                implementation_sha256=selection.implementation_sha256,
                role=ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER,
                required_output_schema_ids=frozenset((edge.codomain_schema,)),
                required_config_schema_id=edge.config.object_schema,
            )
            static_selection = _exact_factory_binding(
                self.executable_factories,
                capability_key=selection.capability_key,
                capability_version=selection.capability_version,
                implementation_sha256=selection.implementation_sha256,
                role=ExecutableBindingRole.SOURCE_PROVIDER,
                required_output_schema_ids=frozenset((edge.codomain_schema,)),
                required_profile_schema_id=SourcePipelineProfile.SCHEMA,
                required_config_schema_id=edge.config.object_schema,
            )
            if runtime_binding is None and static_selection is None:
                missing.append(f"transform-edge.{edge.edge_id}")
        return tuple(sorted(missing))

    def compile(
        self,
        *,
        profile: SourcePipelineProfile,
        transformation_manifest: DatasetTransformationManifest,
        dataset_operation_request: DatasetOperationRequest | None,
    ) -> SourcePipelineCompilation:
        if profile.dataset_transformation_manifest != ObjectIdentity.from_record(
            transformation_manifest.manifest_id,
            transformation_manifest,
        ):
            raise ValueError("source profile binds another transformation manifest")
        matches = tuple(value for value in self.bindings if value.matches(profile))
        if len(matches) != 1:
            raise ValueError("source profile does not resolve to one installed compiler binding")
        missing = self._missing_executable_bindings(profile)
        if missing:
            raise ExecutableBindingRequiredError(missing)
        binding = matches[0]
        return compile_source_pipeline(
            profile,
            source_manifest=binding.source_manifest,
            capability_registry=binding.capability_registry,
            dataset_registry=binding.dataset_registry,
            dataset_operation_request=dataset_operation_request,
        )


@dataclass(frozen=True, slots=True)
class LinkedCampaignProfileCompiler:
    """Installed dependencies for the authority-free linked-profile join."""

    evidence_resolver: EvidenceProfileRegistryResolver
    capability_registry: CapabilityRegistry
    executable_factories: ExecutableCapabilityProviderFactoryRegistry
    conditional_successor_resolver: ObjectIdentity
    public_route: ObjectIdentity

    def __post_init__(self) -> None:
        if not isinstance(
            self.executable_factories,
            ExecutableCapabilityProviderFactoryRegistry,
        ):
            raise TypeError("linked compiler requires the authenticated factory registry")
        if self.public_route.object_schema != 'empirical-lawhood/api/public-composition':
            raise ValueError("linked compiler must bind the current public composition")

    def compile(
        self,
        *,
        profile: LinkedCampaignProfile,
        source_compilation: SourcePipelineCompilation,
        evidence_selection: EvidenceProfileSelection,
        candidate_reports: tuple[ExecutableStudyCompilationReport, ...],
    ) -> LinkedCampaignExecutableCompilation:
        return compile_linked_campaign_profile(
            profile=profile,
            source_compilation=source_compilation,
            evidence_selection=evidence_selection,
            evidence_resolver=self.evidence_resolver,
            candidate_reports=candidate_reports,
            capability_registry=self.capability_registry,
            executable_factories=self.executable_factories,
            conditional_successor_resolver=self.conditional_successor_resolver,
            public_route=self.public_route,
        )


@dataclass(frozen=True, slots=True)
class LinkedCampaignPackageCompilationBinding(CanonicalRecord):
    "Exact topology-node to ExecutableStudyCandidate join for one linked package role."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/linked-campaign-package-compilation-binding'

    binding_id: str
    node: LinkedCampaignPackageNode
    role: LinkedCampaignPackageRole
    base_authoring_package: ObjectIdentity
    candidate_compilation: ObjectIdentity
    candidate: ObjectIdentity | None
    issue_scope_id: str
    execution_scope_id: str
    parent_node_id: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        for name in ("issue_scope_id", "execution_scope_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.parent_node_id is not None:
            validate_stable_id(self.parent_node_id, field_name="parent_node_id")
        if (
            self.role is not self.node.role
            or self.base_authoring_package != self.node.authoring_package
            or self.issue_scope_id != self.node.issue_scope_id
            or self.execution_scope_id != self.node.execution_scope_id
            or self.parent_node_id != self.node.parent_node_id
            or self.candidate_compilation.object_schema
            != ExecutableStudyCompilationReport.SCHEMA
        ):
            raise ValueError("linked package binding differs from its topology node")
        if self.candidate is not None and (
            self.candidate.object_schema != ExecutableStudyCandidate.SCHEMA
        ):
            raise ValueError("linked package binding names another candidate schema")


@dataclass(frozen=True, slots=True)
class LinkedCampaignExecutableCompilation(CanonicalRecord):
    """Authority-free public join over four compiled linked packages."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/linked-campaign-executable-compilation'

    compilation_id: str
    linked_profile: ObjectIdentity
    source_compilation: ObjectIdentity
    evidence_profile_selection: ObjectIdentity
    evidence_registry: ObjectIdentity
    executable_binding_aggregate: ObjectIdentity
    package_bindings: tuple[LinkedCampaignPackageCompilationBinding, ...]
    parent_protocol: ObjectIdentity
    conditional_successor_resolver: ObjectIdentity
    public_route: ObjectIdentity
    provider_registry_fingerprints: tuple[str, ...]
    disposition: LinkedCampaignExecutableDisposition
    reason_codes: tuple[str, ...]
    grants_authority: bool
    executed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.compilation_id, field_name="compilation_id")
        if (
            self.linked_profile.object_schema != LinkedCampaignProfile.SCHEMA
            or self.source_compilation.object_schema != SourcePipelineCompilation.SCHEMA
            or self.evidence_profile_selection.object_schema != EvidenceProfileSelection.SCHEMA
            or self.evidence_registry.object_schema != EvidenceWorldProfileRegistry.SCHEMA
            or self.executable_binding_aggregate.object_schema
            != GeneratedExecutableBindingAggregate.SCHEMA
            or self.parent_protocol.object_schema != ProtocolTemplate.SCHEMA
            or self.public_route.object_schema != 'empirical-lawhood/api/public-composition'
        ):
            raise ValueError("linked compilation binds another profile topology")
        require_sorted_unique_ids(
            self.package_bindings,
            attribute="binding_id",
            field_name="package_bindings",
        )
        if {value.role for value in self.package_bindings} != set(LinkedCampaignPackageRole):
            raise ValueError("linked compilation package roster is incomplete")
        require_sorted_unique_strings(
            self.provider_registry_fingerprints,
            field_name="provider_registry_fingerprints",
            allow_empty=False,
        )
        for value in self.provider_registry_fingerprints:
            validate_sha256(value, field_name="provider_registry_fingerprint")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        positive = (
            self.disposition is LinkedCampaignExecutableDisposition.COMPILED_AUTHORITY_PENDING
        )
        if positive == bool(self.reason_codes):
            raise ValueError("linked compilation disposition/reasons are inconsistent")
        if positive and any(value.candidate is None for value in self.package_bindings):
            raise ValueError("positive linked compilation contains an uncompiled candidate")
        if self.grants_authority or self.executed:
            raise ValueError("profile compilation cannot grant authority or execute")


def _base_candidate(
    report: ExecutableStudyCompilationReport,
) -> DraftStudyCandidate | None:
    if report.candidate is None:
        return None
    return report.candidate.base_candidate.base_candidate


def _package_binding(
    node: LinkedCampaignPackageNode,
    report: ExecutableStudyCompilationReport,
) -> LinkedCampaignPackageCompilationBinding:
    standard = _base_candidate(report)
    base_package = report.base_report.authoring_package
    if base_package != node.authoring_package:
        raise ValueError("linked candidate base authoring package differs from its node")
    if standard is None:
        candidate = None
    else:
        assert report.candidate is not None
        candidate = ObjectIdentity.from_record(report.candidate.candidate_id, report.candidate)
    return LinkedCampaignPackageCompilationBinding(
        binding_id=f"linked-package-binding.{node.node_id}",
        node=node,
        role=node.role,
        base_authoring_package=base_package,
        candidate_compilation=ObjectIdentity.from_record(report.report_id, report),
        candidate=candidate,
        issue_scope_id=node.issue_scope_id,
        execution_scope_id=node.execution_scope_id,
        parent_node_id=node.parent_node_id,
    )


def compile_linked_campaign_profile(
    *,
    profile: LinkedCampaignProfile,
    source_compilation: SourcePipelineCompilation,
    evidence_selection: EvidenceProfileSelection,
    evidence_resolver: EvidenceProfileRegistryResolver,
    candidate_reports: tuple[ExecutableStudyCompilationReport, ...],
    capability_registry: CapabilityRegistry,
    executable_factories: ExecutableCapabilityProviderFactoryRegistry,
    conditional_successor_resolver: ObjectIdentity,
    public_route: ObjectIdentity,
) -> LinkedCampaignExecutableCompilation:
    """Join four precompiled candidates without issuing conditional descendants."""

    if profile.source_readiness != ObjectIdentity.from_record(
        source_compilation.compilation_id,
        source_compilation,
    ):
        raise ValueError("linked profile binds another source compilation")
    if profile.source_profile != source_compilation.profile:
        raise ValueError("linked profile/source profile identities differ")
    selection_identity = ObjectIdentity.from_record(
        evidence_selection.selection_id, evidence_selection
    )
    if profile.evidence_profile_selection != selection_identity:
        raise ValueError("linked profile binds another evidence selection")
    feasibility = resolve_candidate_profile_feasibility(
        selection=evidence_selection,
        resolver=evidence_resolver,
    )
    reasons: set[str] = set()
    if source_compilation.disposition is SourcePipelineCompilationDisposition.AUTHORITY_REQUIRED:
        disposition = LinkedCampaignExecutableDisposition.SOURCE_AUTHORITY_REQUIRED
        reasons.update(source_compilation.reason_codes)
    elif (
        feasibility.disposition is not ProfileFeasibilityDisposition.READY_FOR_CANDIDATE_COMPILATION
    ):
        disposition = LinkedCampaignExecutableDisposition.EVIDENCE_PROFILE_BLOCKED
        reasons.update(value.value for value in feasibility.reason_codes)
    else:
        disposition = LinkedCampaignExecutableDisposition.COMPILED_AUTHORITY_PENDING

    if len(candidate_reports) != len(LinkedCampaignPackageRole):
        raise ValueError("linked profile requires exactly four candidate reports")
    report_ids = tuple(value.report_id for value in candidate_reports)
    if len(set(report_ids)) != len(report_ids):
        raise ValueError("linked candidate reports repeat one report identity")
    reports_by_base: dict[ObjectIdentity, ExecutableStudyCompilationReport] = {}
    for report in candidate_reports:
        standard = _base_candidate(report)
        base = report.base_report.authoring_package
        if base in reports_by_base:
            raise ValueError("linked candidate reports repeat one base authoring package")
        reports_by_base[base] = report
        if (
            standard is None
            or report.disposition is not CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING
        ):
            if disposition is LinkedCampaignExecutableDisposition.COMPILED_AUTHORITY_PENDING:
                disposition = LinkedCampaignExecutableDisposition.CANDIDATE_COMPILATION_BLOCKED
            reasons.add("CANDIDATE_NOT_COMPILED_AUTHORITY_PENDING")
            continue
        assert report.candidate is not None
        candidate_base = ObjectIdentity.from_record(
            report.candidate.authoring_package.base.package_id,
            report.candidate.authoring_package.base,
        )
        if candidate_base != base:
            raise ValueError("linked candidate report/base package identity differs")
        programme = standard
        if (
            ObjectIdentity.from_record(programme.system.system_id, programme.system)
            != profile.system
            or ObjectIdentity.from_record(programme.experiment.experiment_id, programme.experiment)
            != profile.experiment
            or ObjectIdentity.from_record(programme.campaign.campaign_id, programme.campaign)
            != profile.campaign
        ):
            raise ValueError("linked candidate system/experiment/campaign identity differs")

    package_bindings: list[LinkedCampaignPackageCompilationBinding] = []
    for node in profile.topology.nodes:
        matched_report = reports_by_base.get(node.authoring_package)
        if matched_report is None:
            raise ValueError("linked topology node lacks one exact candidate report")
        package_bindings.append(_package_binding(node, matched_report))

    required_executables: set[tuple[str, str, str, str, frozenset[str], str | None]] = set()
    for value in profile.capabilities:
        selection = value.selection
        try:
            manifest = capability_registry.resolve(
                selection.capability_key,
                selection.capability_version,
            )
        except KeyError:
            raise ExecutableBindingRequiredError((value.binding_id,)) from None
        if manifest.implementation_sha256 != selection.implementation_sha256:
            raise ValueError("linked campaign capability implementation drifted")
        required_executables.add(
            (
                value.binding_id,
                selection.capability_key,
                selection.capability_version,
                selection.implementation_sha256,
                frozenset((LinkedCampaignStageEnvelope.SCHEMA,)),
                value.config_schema,
            )
        )
    for report in candidate_reports:
        candidate = _base_candidate(report)
        if candidate is None:
            continue
        for selection in candidate.capability_locks:
            required_outputs = frozenset(
                output.payload_schema
                for step in candidate.protocol.steps
                if step.capability_key == selection.capability_key
                and step.capability_version == selection.capability_version
                for output in step.outputs
            )
            if not required_outputs:
                raise ValueError("candidate capability lock owns no protocol output")
            try:
                manifest = capability_registry.resolve(
                    selection.capability_key,
                    selection.capability_version,
                )
            except KeyError:
                reasons.add(
                    f"EXECUTABLE_BINDING_REQUIRED:candidate-capability.{selection.selection_id}"
                )
                if disposition is LinkedCampaignExecutableDisposition.COMPILED_AUTHORITY_PENDING:
                    disposition = LinkedCampaignExecutableDisposition.EXECUTABLE_BINDING_REQUIRED
                continue
            if (
                manifest.implementation_sha256 != selection.implementation_sha256
                or not required_outputs.issubset(manifest.output_schema_ids)
            ):
                raise ValueError("candidate capability differs from the generated registry")
            config_schemas = {
                step.config.config_schema
                for step in candidate.protocol.steps
                if step.capability_key == selection.capability_key
                and step.capability_version == selection.capability_version
            }
            if len(config_schemas) != 1:
                raise ValueError("candidate capability has ambiguous protocol config schemas")
            required_executables.add(
                (
                    f"candidate-capability.{selection.selection_id}",
                    selection.capability_key,
                    selection.capability_version,
                    selection.implementation_sha256,
                    required_outputs,
                    next(iter(config_schemas)),
                )
            )
    missing_bindings = tuple(
        binding_id
        for (
            binding_id,
            capability_key,
            capability_version,
            implementation_sha256,
            output_schema_ids,
            config_schema_id,
        ) in sorted(required_executables, key=lambda value: value[:4])
        if _exact_factory_binding(
            executable_factories,
            capability_key=capability_key,
            capability_version=capability_version,
            implementation_sha256=implementation_sha256,
            role=ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER,
            required_output_schema_ids=output_schema_ids,
            required_config_schema_id=config_schema_id,
        )
        is None
    )
    if missing_bindings:
        if disposition is LinkedCampaignExecutableDisposition.COMPILED_AUTHORITY_PENDING:
            disposition = LinkedCampaignExecutableDisposition.EXECUTABLE_BINDING_REQUIRED
        reasons.update(f"EXECUTABLE_BINDING_REQUIRED:{value}" for value in missing_bindings)

    protocol = build_linked_campaign_protocol(profile, capability_registry)
    protected_parent = next(
        value
        for value in profile.topology.nodes
        if value.role is LinkedCampaignPackageRole.PROTECTED_MEASUREMENT_THROUGH_ADMISSION_PARENT
    )
    protected_report = reports_by_base[protected_parent.authoring_package]
    protected_candidate = _base_candidate(protected_report)
    if protected_candidate is not None and protected_candidate.protocol != protocol:
        raise ValueError("protected-parent candidate protocol differs from linked profile")
    registry_fingerprint = capability_registry.fingerprint()
    executable_aggregate = executable_factories.aggregate
    executable_identity = ObjectIdentity.from_record(
        executable_aggregate.aggregate_id,
        executable_aggregate,
    )
    evidence_registry = evidence_resolver.resolve_registry(evidence_selection)
    if disposition is LinkedCampaignExecutableDisposition.COMPILED_AUTHORITY_PENDING:
        reasons.clear()
    elif not reasons:
        reasons.add(disposition.value)
    linked_identity = ObjectIdentity.from_record(profile.profile_id, profile)
    source_identity = ObjectIdentity.from_record(
        source_compilation.compilation_id,
        source_compilation,
    )
    evidence_registry_identity = ObjectIdentity.from_record(
        evidence_registry.registry_id,
        evidence_registry,
    )
    ordered_bindings = tuple(sorted(package_bindings, key=lambda value: value.binding_id))
    reason_codes = tuple(sorted(reasons))
    provider_registry_fingerprints = (registry_fingerprint,)
    compilation_digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "linked_profile": linked_identity,
                "source_compilation": source_identity,
                "evidence_profile_selection": selection_identity,
                "evidence_registry": evidence_registry_identity,
                "executable_binding_aggregate": executable_identity,
                "package_bindings": ordered_bindings,
                "parent_protocol": ObjectIdentity.from_record(protocol.template_id, protocol),
                "conditional_successor_resolver": conditional_successor_resolver,
                "public_route": public_route,
                "provider_registry_fingerprints": provider_registry_fingerprints,
                "disposition": disposition,
                "reason_codes": reason_codes,
            }
        )
    ).hexdigest()
    return LinkedCampaignExecutableCompilation(
        compilation_id=f"linked-executable-compilation.{compilation_digest[:32]}",
        linked_profile=linked_identity,
        source_compilation=source_identity,
        evidence_profile_selection=selection_identity,
        evidence_registry=evidence_registry_identity,
        executable_binding_aggregate=executable_identity,
        package_bindings=ordered_bindings,
        parent_protocol=ObjectIdentity.from_record(protocol.template_id, protocol),
        conditional_successor_resolver=conditional_successor_resolver,
        public_route=public_route,
        provider_registry_fingerprints=provider_registry_fingerprints,
        disposition=disposition,
        reason_codes=reason_codes,
        grants_authority=False,
        executed=False,
    )


__all__ = [
    "ExecutableBindingRequiredError",
    'LinkedCampaignExecutableCompilation',
    'LinkedCampaignExecutableDisposition',
    'LinkedCampaignPackageCompilationBinding',
    'LinkedCampaignProfileCompiler',
    'SourceProfileCompilerBinding',
    'SourceProfileCompilerRegistry',
    "compile_linked_campaign_profile",
]
