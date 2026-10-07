# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
"""Truth-known issued-study fixture with separate extension approval."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Protocol

from empirical_lawhood.api.facade import EmpiricalLawhoodApi
from empirical_lawhood.api.models import IssuedStudyPackage, EnvelopeExperimentPackage, assemble_issued_study_package, assemble_envelope_experiment_package
from empirical_lawhood.api.results import OperationStatus
from empirical_lawhood.kernel.authority import AuthorityAction, SourceAccessClass
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.models import ViewModelSetSpec, ModelSetSpec
from empirical_lawhood.planning.adaptive_acquisition import (
    FixedRoundAcquisitionPlan,
    FixedRoundArmSpec,
    FixedRoundStopRule,
)
from empirical_lawhood.planning.approval import (
    ApprovalCheckerRegistry,
    AttestationResult,
    CompleteApprovalService,
    issue_approval_gate_attestation,
)
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority
from empirical_lawhood.runtime.artifacts import (
    ArtifactGenericValidation,
    ArtifactMaterialization,
    ArtifactProfile,
    ArtifactPublicationBinding,
    ArtifactPublicationMember,
    ArtifactPublicationScope,
    LogicalArtifactIdentity,
    artifact_publication_batch_id,
    artifact_publication_commit_relative_path,
    artifact_publication_member,
)
from empirical_lawhood.runtime.candidate_compiler import bind_standard_candidate_extensions
from empirical_lawhood.runtime.execution_envelope import (
    ExecutionEnvelopeCellSpec,
    ExecutionEnvelopeSpec,
)
from empirical_lawhood.runtime.study_issue import IssuedStudyMember, StudyPublicationReceipt, StudyExtensionDecoderRegistration, IssuedExecutableStudyManifest, bind_standard_issued_extensions, build_study_approval_proposal, validate_issued_study_extensions
from tests.approval_support import FixedDecisionClock, deterministic_approval_signer
from tests.runtime_platform.test_study_approval import _AttestationStore, _AuthorizationStore, _StudyProposalStore, _checker_registrations
from tests.runtime_platform.test_study_issue import _issue_api_fixture, _issue_request


def _digest(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode()
    return hashlib.sha256(payload).hexdigest()


def _identity(object_id: str, schema: str) -> ObjectIdentity:
    return ObjectIdentity(
        object_id=object_id,
        object_schema=schema,
        object_version="1.0.0",
        object_fingerprint=_digest(object_id),
    )


@dataclass(frozen=True, slots=True)
class IssuedStudyFixture:
    base: IssuedStudyPackage
    package: EnvelopeExperimentPackage
    approval_service: CompleteApprovalService


class IssuedStudyIssueFixtureFactory(Protocol):
    def __call__(
        self,
        tmp_path: Path,
    ) -> tuple[EmpiricalLawhoodApi, dict[str, Path], Path, StudyOperationAuthority]: ...


def _approval_service(system_id: str, required_gate_ids: tuple[str, ...]):
    proposal_store = _StudyProposalStore()
    authorization_store = _AuthorizationStore()
    attestation_store = _AttestationStore()
    registrations = _checker_registrations(required_gate_ids)
    service = CompleteApprovalService(
        clock=FixedDecisionClock(
            clock_id=f"issued-study-approval-clock.{system_id}",
            value="2026-08-22T20:40:00Z",
        ),
        proposal_store=proposal_store,
        checker_registry=ApprovalCheckerRegistry(
            registry_id=f"issued-study-approval-checkers.{system_id}",
            registrations=registrations,
        ),
        attestation_store=attestation_store,
        store=authorization_store,
    )
    return service, proposal_store, attestation_store, registrations


def _authorize(
    *,
    service: CompleteApprovalService,
    proposal_store: _StudyProposalStore,
    attestation_store: _AttestationStore,
    registrations,  # type: ignore[no-untyped-def]
    proposal,  # type: ignore[no-untyped-def]
    system,  # type: ignore[no-untyped-def]
    implementation_commit: str,
    authorization_id: str,
):  # type: ignore[no-untyped-def]
    frozen_identity = service.freeze_study_proposal(
        proposal=proposal,
        system=system,
        requested_scope_id="reference-world",
        implementation_commit=implementation_commit,
        authority_action=AuthorityAction.REFERENCE_WORLD_EXECUTION,
        source_access=SourceAccessClass.NONE,
    )
    for registration in registrations:
        attestation = issue_approval_gate_attestation(
            registration=registration,
            signer=deterministic_approval_signer(registration.gate_id),
            attestation_id=f"attestation.{registration.gate_id}.{authorization_id}",
            authorization_id=authorization_id,
            subject=frozen_identity,
            result=AttestationResult.PASSED,
            checked_at_utc="2026-08-22T20:35:00Z",
            issued_at_utc="2026-08-22T20:36:00Z",
            information_cutoff=proposal.decision_cutoff,
        )
        attestation_store.records[attestation.attestation_id] = attestation
    approval = service.authorize(
        policy=system.authority_policy,
        frozen_proposal=frozen_identity,
        attestations=tuple(
            ObjectIdentity.from_record(value.attestation_id, value)
            for value in sorted(
                (
                    item
                    for item in attestation_store.records.values()
                    if item.authorization_id == authorization_id
                ),
                key=lambda value: value.gate_id,
            )
        ),
        authorization_id=authorization_id,
        approver_id=system.authority_policy.delegate_id,
    )
    return proposal_store.load(frozen_identity.object_id), approval


def _extension_plan() -> FixedRoundAcquisitionPlan:
    return FixedRoundAcquisitionPlan(
        acquisition_plan_id="acquisition.fixed-round-fixture",
        arms=(
            FixedRoundArmSpec(
                arm_id="arm.fixed-round-fixture",
                query=_identity("query.fixed-round-fixture", 'empirical-lawhood/testing/query'),
                maximum_selections=1,
            ),
        ),
        maximum_rounds=1,
        selections_per_round=1,
        selector_capability_key="fixed-round-fixture-selector",
        selector_capability_version="1.0.0",
        selector_config_sha256=_digest("fixed-round-fixture-selector-config"),
        selector_implementation_sha256=_digest("fixed-round-fixture-selector-implementation"),
        stop_rule=FixedRoundStopRule.EXHAUST_ROUNDS,
        checkpoint_ids=("checkpoint.fixed-round-001",),
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def _issued_study_publication(
    manifest: IssuedExecutableStudyManifest,
) -> StudyPublicationReceipt:
    storage_root_id = manifest.base.storage_root_id
    scope = ArtifactPublicationScope(
        publication_scope_id=f"publication-scope.{manifest.issue_id}",
        storage_root_id=storage_root_id,
        relative_root=".",
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    member_publications = tuple(
        ArtifactPublicationMember(
            materialization_id=f"materialization.{member.member_id}",
            logical_artifact_id=member.member_id,
            logical_identity_sha256=_digest(f"logical.{member.member_id}"),
            storage_root_id=storage_root_id,
            relative_path=member.relative_path,
            physical_sha256=member.physical_sha256,
            size_bytes=member.size_bytes,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        for member in manifest.members
    )
    manifest_id = f"issued-manifest.{manifest.issue_id}"
    manifest_payload = manifest.canonical_bytes()
    manifest_logical = LogicalArtifactIdentity(
        logical_artifact_id=manifest_id,
        content_sha256=manifest.fingerprint(),
        payload_schema=manifest.SCHEMA,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type="application/json",
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        parent_visibility_ceilings=(),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        generic_validation=ArtifactGenericValidation(
            validator_key="canonical-json-issued-study-fixture",
            validator_version="1.0.0",
            validator_implementation_sha256=_digest("canonical-json-issued-study-fixture"),
            payload_schema=manifest.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON,
        ),
    )
    manifest_materialization = ArtifactMaterialization(
        materialization_id=f"materialization.{manifest_id}",
        logical_artifact_id=manifest_id,
        storage_root_id=storage_root_id,
        relative_path=f"issued-programmes/{manifest.issue_id}/manifest.json",
        physical_sha256=_digest(manifest_payload),
        size_bytes=len(manifest_payload),
        compression="none",
    )
    members = tuple(
        sorted(
            (
                *member_publications,
                artifact_publication_member(
                    manifest_logical,
                    manifest_materialization,
                ),
            ),
            key=lambda value: value.materialization_id,
        )
    )
    batch_id = artifact_publication_batch_id(scope, members)
    publication = ArtifactPublicationBinding(
        publication_batch_id=batch_id,
        publication_scope=scope,
        commit_marker_relative_path=artifact_publication_commit_relative_path(
            scope,
            batch_id,
        ),
        commit_marker_sha256=_digest(f"commit.{batch_id}"),
        commit_marker_size_bytes=1,
        members=members,
    )
    return StudyPublicationReceipt(
        receipt_id=f"issue-publication-receipt.{manifest.issue_id}",
        issue_manifest=ObjectIdentity.from_record(manifest.issue_id, manifest),
        manifest_logical=manifest_logical,
        manifest_materialization=manifest_materialization,
        publication=publication,
        declared_member_ids=tuple(value.member_id for value in manifest.members),
        total_payload_bytes=len(manifest_payload)
        + sum(value.size_bytes for value in manifest.members),
        published_at_utc="2026-08-22T20:30:00Z",
    )


def build_issued_study_fixture(
    tmp_path: Path,
    *,
    issue_fixture_factory: IssuedStudyIssueFixtureFactory = _issue_api_fixture,
    model_set: ViewModelSetSpec | ModelSetSpec | None = None,
    additional_extensions: tuple[
        tuple[CanonicalRecord, StudyExtensionDecoderRegistration], ...
    ] = (),
    include_fixed_round_extension: bool = True,
) -> IssuedStudyFixture:
    issue_api, paths, _external, _custody = issue_fixture_factory(tmp_path)
    issued_result = issue_api.issue_study(_issue_request(paths, confirmed=True))
    assert issued_result.status is OperationStatus.SUCCEEDED
    assert issued_result.payload is not None
    assert issued_result.payload.publication_receipt is not None
    base_manifest = issued_result.payload.manifest
    base_publication = issued_result.payload.publication_receipt
    base_candidate = base_manifest.candidate.base_candidate
    service, proposal_store, attestation_store, registrations = _approval_service(
        base_candidate.system.system_id,
        base_candidate.system.authority_policy.required_gate_ids,
    )
    base_proposal = build_study_approval_proposal(
        issued_study=base_manifest,
        publication_receipt=base_publication,
    )
    base_frozen, base_approval = _authorize(
        service=service,
        proposal_store=proposal_store,
        attestation_store=attestation_store,
        registrations=registrations,
        proposal=base_proposal,
        system=base_candidate.system,
        implementation_commit=base_manifest.implementation_commit,
        authorization_id="authorization.fixture-base",
    )
    base_execution = StudyOperationAuthority(
        authority_id="authority.execution.fixture-base",
        kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
        subject=ObjectIdentity.from_record(base_manifest.issue_id, base_manifest),
        prerequisite_authority=ObjectIdentity.from_record(
            base_approval.authorization_id,
            base_approval,
        ),
        issuer=base_manifest.proposer_attestation.proposer,
        grantee_id="operator.execution-service",
        scope_id="scope.fixture-base-execution",
        storage_root_id=None,
        relative_root=None,
        allows_source_acquisition=False,
        allows_external_publication=False,
        allows_execution=True,
        allows_actuation=False,
        allows_reveal=False,
        issued_at_utc="2026-08-22T20:41:00Z",
        expires_at_utc="2026-08-23T20:41:00Z",
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
    )
    base = assemble_issued_study_package(
        issued_study=base_manifest,
        publication_receipt=base_publication,
        frozen_proposal=base_frozen,
        scientific_approval=base_approval,
        execution_authority=base_execution,
        run_plan_id="run.fixture-base",
        grantee_id="operator.execution-service",
        at_utc="2026-08-22T20:42:00Z",
        model_set=model_set,
    )

    extension_payload = _extension_plan().canonical_bytes()
    fixed_proposal = ProposedStudyExtension(
        extension_id="extension.fixture-fixed-round",
        namespace_id="fixed-round-acquisition",
        payload=ObjectIdentity.from_record("acquisition.fixed-round-fixture", _extension_plan()),
        payload_size_bytes=len(extension_payload),
        decoder_key="fixed-round-fixture-decoder",
        decoder_version="1.0.0",
        decoder_config_sha256=_digest("fixed-round-fixture-decoder-config"),
        required_for_activation=True,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    fixed_member = IssuedStudyMember(
        member_id="issued-member.fixture-fixed-round",
        relative_path="issued-programmes/extensions-fixture/fixed-round.json",
        payload_schema=FixedRoundAcquisitionPlan.SCHEMA,
        media_type="application/json",
        profile=ArtifactProfile.CANONICAL_JSON,
        size_bytes=len(extension_payload),
        physical_sha256=_digest(extension_payload),
        logical_content_sha256=_extension_plan().fingerprint(),
    )
    fixed_decoder = StudyExtensionDecoderRegistration(
        registration_id="decoder-registration.fixture-fixed-round",
        decoder_key="fixed-round-fixture-decoder",
        decoder_version="1.0.0",
        payload_schema=FixedRoundAcquisitionPlan.SCHEMA,
        payload_version=FixedRoundAcquisitionPlan.VERSION,
        config_sha256=_digest("fixed-round-fixture-decoder-config"),
        implementation_sha256=_digest("fixed-round-fixture-decoder-implementation"),
        maximum_payload_bytes=1_000_000,
    )
    extra_proposals = []
    extra_members = []
    extra_decoders = []
    for index, (record, registration) in enumerate(additional_extensions, start=1):
        payload = record.canonical_bytes()
        record_id = next(
            (
                getattr(record, name)
                for name in (
                    "spec_id",
                    "plan_id",
                    "profile_id",
                    "record_id",
                    "extension_set_id",
                    "binding_id",
                    "config_id",
                    "manifest_id",
                    "topology_id",
                )
                if isinstance(getattr(record, name, None), str)
            ),
            None,
        )
        if record_id is None:
            raise ValueError("additional issued-study fixture extension lacks a stable record id")
        suffix = f"{index:02d}-{registration.decoder_key}"
        extra_proposals.append(
            ProposedStudyExtension(
                extension_id=f"extension.fixture-{suffix}",
                namespace_id=f"installed-method-config-{index:02d}",
                payload=ObjectIdentity.from_record(record_id, record),
                payload_size_bytes=len(payload),
                decoder_key=registration.decoder_key,
                decoder_version=registration.decoder_version,
                decoder_config_sha256=registration.config_sha256,
                required_for_activation=True,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            )
        )
        extra_members.append(
            IssuedStudyMember(
                member_id=f"issued-member.fixture-{suffix}",
                relative_path=f"issued-programmes/extensions-fixture/{suffix}.json",
                payload_schema=record.SCHEMA,
                media_type="application/json",
                profile=ArtifactProfile.CANONICAL_JSON,
                size_bytes=len(payload),
                physical_sha256=_digest(payload),
                logical_content_sha256=record.fingerprint(),
            )
        )
        extra_decoders.append(registration)
    proposed = ProposedStudyExtensionSet(
        extension_set_id="extensions.issued-fixture",
        authoring_package=ObjectIdentity.from_record(
            base_manifest.authoring_package.package_id,
            base_manifest.authoring_package,
        ),
        namespace_roster=tuple(
            sorted(
                {
                    *(("fixed-round-acquisition",) if include_fixed_round_extension else ()),
                    *(value.namespace_id for value in extra_proposals),
                }
            )
        ),
        extensions=tuple(
            sorted(
                (
                    *((fixed_proposal,) if include_fixed_round_extension else ()),
                    *extra_proposals,
                ),
                key=lambda value: value.extension_id,
            )
        ),
    )
    authoring_definition = ExecutableStudyDefinition(
        package_id="authoring-package.issued-fixture",
        base=base_manifest.authoring_package,
        extension_set=proposed,
    )
    extension_candidate = bind_standard_candidate_extensions(
        base_candidate=base_manifest.candidate,
        authoring_package=authoring_definition,
    )
    issued_extensions, validation = validate_issued_study_extensions(
        proposed=proposed,
        issued_members=tuple(
            sorted(
                (
                    *((fixed_member,) if include_fixed_round_extension else ()),
                    *extra_members,
                ),
                key=lambda value: value.member_id,
            )
        ),
        decoder_registrations=tuple(
            sorted(
                (
                    *((fixed_decoder,) if include_fixed_round_extension else ()),
                    *extra_decoders,
                ),
                key=lambda value: value.registration_id,
            )
        ),
    )
    extension_attestation = replace(
        base_manifest.proposer_attestation,
        attestation_id="attestation.fixture-extensions",
        candidate=ObjectIdentity.from_record(extension_candidate.candidate_id, extension_candidate),
        raw_materialization_sha256=authoring_definition.fingerprint(),
    )
    extension_custody = StudyOperationAuthority(
        authority_id="authority.custody.fixture-extensions",
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        subject=ObjectIdentity.from_record(extension_candidate.candidate_id, extension_candidate),
        prerequisite_authority=None,
        issuer=extension_attestation.proposer,
        grantee_id="operator.issue-service",
        scope_id="scope.fixture-extension-publication",
        storage_root_id="external-test",
        relative_root="issued-programmes",
        allows_source_acquisition=False,
        allows_external_publication=True,
        allows_execution=False,
        allows_actuation=False,
        allows_reveal=False,
        issued_at_utc="2026-08-22T20:40:30Z",
        expires_at_utc="2026-08-23T20:40:30Z",
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    issued_manifest = bind_standard_issued_extensions(
        base=base_manifest,
        candidate=extension_candidate,
        issued_extensions=issued_extensions,
        extension_validation=validation,
        extension_proposer_attestation=extension_attestation,
        extension_custody_authority=extension_custody,
    )
    extension_publication = _issued_study_publication(issued_manifest)
    extension_approval_proposal = build_study_approval_proposal(
        issued_study=issued_manifest,
        publication_receipt=extension_publication,
    )
    approved_extension_proposal, extension_approval = _authorize(
        service=service,
        proposal_store=proposal_store,
        attestation_store=attestation_store,
        registrations=registrations,
        proposal=extension_approval_proposal,
        system=base_candidate.system,
        implementation_commit=issued_manifest.implementation_commit,
        authorization_id="authorization.fixture-extensions",
    )
    extension_execution_authority = StudyOperationAuthority(
        authority_id="authority.execution.fixture-extensions",
        kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
        subject=ObjectIdentity.from_record(issued_manifest.issue_id, issued_manifest),
        prerequisite_authority=ObjectIdentity.from_record(
            extension_approval.authorization_id,
            extension_approval,
        ),
        issuer=issued_manifest.proposer_attestation.proposer,
        grantee_id="operator.execution-service",
        scope_id="scope.fixture-extension-execution",
        storage_root_id=None,
        relative_root=None,
        allows_source_acquisition=False,
        allows_external_publication=False,
        allows_execution=True,
        allows_actuation=False,
        allows_reveal=False,
        issued_at_utc="2026-08-22T20:43:00Z",
        expires_at_utc="2026-08-23T20:43:00Z",
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
    )
    envelope = ExecutionEnvelopeSpec(
        envelope_spec_id="execution-envelope-spec.issued-fixture",
        issued_study_extensions=ObjectIdentity.from_record(
            issued_extensions.issued_extension_set_id,
            issued_extensions,
        ),
        cells=tuple(
            ExecutionEnvelopeCellSpec(
                cell_id=f"cell.{step.step_id}",
                task_id=step.step_id,
                physical_independent_unit_id=(
                    f"{base_candidate.system.independent_unit.unit_id}.{step.step_id}"
                ),
                maximum_physical_tokens=step.maximum_attempts,
                maximum_retry_tokens=step.maximum_attempts - 1,
            )
            for step in base_candidate.protocol.steps
        ),
        global_physical_token_limit=sum(
            value.maximum_attempts for value in base_candidate.protocol.steps
        ),
        global_retry_token_limit=sum(
            value.maximum_attempts - 1 for value in base_candidate.protocol.steps
        ),
        allowlisted_retry_reason_codes=("TASK_EXECUTION_FAILED",),
        deadline_utc="2026-08-23T20:40:00Z",
        resource_contract_sha256=_digest("issued-study-fixture-resource-contract"),
        nonrefundable_reservations=True,
        first_valid_success_wins=True,
        unknown_completion_requires_new_disposition=True,
    )
    package = assemble_envelope_experiment_package(
        base=base,
        issued_study=issued_manifest,
        publication_receipt=extension_publication,
        frozen_proposal=approved_extension_proposal,
        scientific_approval=extension_approval,
        execution_authority=extension_execution_authority,
        execution_envelope_spec=envelope,
        run_plan_id="run.fixture-extensions",
        grantee_id="operator.execution-service",
        at_utc="2026-08-22T20:44:00Z",
    )
    return IssuedStudyFixture(base=base, package=package, approval_service=service)


__all__ = [
    "IssuedStudyFixture",
    "IssuedStudyIssueFixtureFactory",
    'build_issued_study_fixture',
]
