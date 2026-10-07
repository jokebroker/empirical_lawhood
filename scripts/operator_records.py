# SPDX-License-Identifier: MPL-2.0
"""Explicit record handoffs for the operator walkthrough; no approval decisions.

Adapted from the source project's programme operator support. No standing grant,
home-directory trust default, signing key, automatic PASSED attestation or native
launch is supplied here. Callers provide each independently reviewed record.
These checkout helpers are examples; the versioned records and API are the
application boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from empirical_lawhood.api.composition import (
    OperatorApprovalRuntimeComposition,
    compose_operator_approval_runtime,
)
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.infrastructure.bounded_io import BoundedFileIOError, read_bounded_bytes
from empirical_lawhood.infrastructure.study_issue import ExternalIssuedStudyPublisher, ExternalStudyOperationAuthorityStore, PROGRAMME_ISSUE_GRANTEE_ID
from empirical_lawhood.kernel.authority import AuthorityAction, SourceAccessClass
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.approval import ApprovalGateAttestation, DurableAuthorizationRecord, FrozenIssuedStudyApprovalProposal
from empirical_lawhood.planning.study_issue import StudyOperationAuthority
from empirical_lawhood.runtime.operator_profile import OperatorStorageAccessMode, OperatorStorageProfile, resolve_external_root_contract
from empirical_lawhood.runtime.study_issue import PublishedStudy, PublishedExecutableStudy, build_study_approval_proposal

R = TypeVar("R", bound=CanonicalRecord)

if TYPE_CHECKING:
    from empirical_lawhood.adapters.composition.reactor_prefix_response.fresh_authoring import AssignedReactorAuthoringProfile


def make_assigned_reactor_profile(
    *, profile_id: str, experiment_id: str, public_source_sha256: str,
    census_id: str, prior_unit_ids: tuple[str, ...], prior_seed_ids: tuple[str, ...],
    source_inventory: ObjectIdentity, noise_seeds: tuple[int, ...], evidence_role: str,
) -> AssignedReactorAuthoringProfile:
    """Construct canonical inputs from explicit owner knowledge; make no review act.

    Census completeness and freshness are human assertions to review separately.
    This checks structure, exact census binding and both native seed/unit aliases.
    It neither draws seeds nor creates authority, signatures or scientific output.
    """
    from empirical_lawhood.adapters.composition.reactor_prefix_response.fresh_authoring import AssignedReactorAuthoringProfile
    from empirical_lawhood.adapters.simulators.reactor_prefix_response.assigned import ReactorPrefixAssignedUnit, ReactorPrefixAssignment, ReactorPrefixPriorCensus
    from empirical_lawhood.adapters.simulators.reactor_prefix_response.panel import SCENARIOS

    if len(noise_seeds) != len(SCENARIOS):
        raise ValueError("exactly five explicit native noise seeds are required")
    census = ReactorPrefixPriorCensus(
        census_id, tuple(sorted(prior_unit_ids)), tuple(sorted(prior_seed_ids)), source_inventory,
    )
    assignment_id = f"{experiment_id}.cohort"
    assignment = ReactorPrefixAssignment(
        assignment_id,
        tuple(ReactorPrefixAssignedUnit(
            f"{assignment_id}.{scenario.replace('_', '-')}", scenario, seed,
        ) for scenario, seed in zip(SCENARIOS, noise_seeds, strict=True)),
        identity(census, census.census_id), evidence_role,
    )
    return AssignedReactorAuthoringProfile(
        profile_id, experiment_id, public_source_sha256, assignment, census,
    )


def make_storage_profile(
    *, profile_id: str, external_root: Path, required_mount: Path,
    expected_mount_source: str | None, expected_volume_identity: str | None,
    filesystem_types: tuple[str, ...], minimum_free_bytes: int,
    access_mode: OperatorStorageAccessMode, maximum_parallel_tasks: int,
) -> OperatorStorageProfile:
    """Construct deployment settings with fixed containment and no authority.

    Run doctor on the actual selected mount before use. This constructor does not
    mount a device, grant access, or claim that a directory satisfies the contract.
    """
    return OperatorStorageProfile(
        profile_id, "external-filesystem", "1.0.0", str(external_root), str(required_mount),
        "artifacts", "scientific-scratch", access_mode, expected_mount_source,
        expected_volume_identity, tuple(sorted(filesystem_types)),
        "strict-mount-contained-no-symlink", minimum_free_bytes, (),
        maximum_parallel_tasks, False,
    )


def read_record(path: Path, record_type: type[R], *, maximum_bytes: int = 256 * 1024**2) -> R:
    return decode_canonical_bytes(read_bounded_bytes(path, maximum_bytes=maximum_bytes),
                                  record_type, maximum_bytes=maximum_bytes)


def export_record(path: Path, record: CanonicalRecord) -> Path:
    """Export canonical operator input without replacing a different prior act.

    The caller selects an existing private work directory. This is an input/export
    file, not an artifact publication or grant installation; use the typed store
    for those operations.
    """
    payload = record.canonical_bytes()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        try:
            retained = read_bounded_bytes(path, maximum_bytes=len(payload))
        except BoundedFileIOError as error:
            raise FileExistsError(f"operator record differs from retained identity: {path}") from error
        if retained != payload:
            raise FileExistsError(f"operator record differs from retained identity: {path}")
    else:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    return path


def identity(record: CanonicalRecord, object_id: str) -> ObjectIdentity:
    return ObjectIdentity.from_record(object_id, record)


@dataclass(frozen=True)
class OperatorRecordStores:
    plane: ExternalArtifactPlane
    authority: ExternalStudyOperationAuthorityStore
    publisher: ExternalIssuedStudyPublisher
    approval: OperatorApprovalRuntimeComposition


def open_stores(*, repo_root: Path, profile: OperatorStorageProfile, trust_path: Path) -> OperatorRecordStores:
    contract = resolve_external_root_contract(profile, repo_root=repo_root, home_root=Path.home())
    plane = ExternalArtifactPlane(GuardedExternalRoot(contract))
    plane.root.verify(for_write=True)
    authority = ExternalStudyOperationAuthorityStore(plane)
    return OperatorRecordStores(
        plane, authority,
        ExternalIssuedStudyPublisher(artifact_plane=plane, authority_store=authority,
                                        grantee_id=PROGRAMME_ISSUE_GRANTEE_ID),
        compose_operator_approval_runtime(artifact_plane=plane, checker_trust_path=trust_path),
    )


def publish_reviewed_authority(stores: OperatorRecordStores, record: StudyOperationAuthority) -> ObjectIdentity:
    """Install only the supplied operator act and verify its immutable replay."""
    expected = stores.authority.persist(record)
    if stores.authority.load(record.authority_id) != record:
        raise RuntimeError("operation authority failed exact replay")
    return expected


def freeze_for_review(
    stores: OperatorRecordStores, *,
    publication: PublishedStudy | PublishedExecutableStudy,
    system: SystemSpec, scope_id: str, implementation_commit: str,
    action: AuthorityAction,
) -> FrozenIssuedStudyApprovalProposal:
    """Freeze the exact issued subject; it still needs signed checker decisions."""
    proposal = build_study_approval_proposal(
        issued_study=publication.manifest, publication_receipt=publication.publication_receipt,
    )
    if action is AuthorityAction.EVALUATOR_REVEAL:
        proposal = replace(proposal, proposal_id=f"{proposal.proposal_id}.reveal")
    frozen_id = stores.approval.service.freeze_study_proposal(
        proposal=proposal, system=system, requested_scope_id=scope_id,
        implementation_commit=implementation_commit, authority_action=action,
        source_access=SourceAccessClass.NONE,
    )
    return stores.approval.study_proposal_store.load(frozen_id.object_id)


def authorize_reviewed(
    stores: OperatorRecordStores, *, system: SystemSpec,
    frozen: FrozenIssuedStudyApprovalProposal, authorization_id: str,
    attestations: tuple[ApprovalGateAttestation, ...], approver_id: str,
) -> DurableAuthorizationRecord:
    """Verify supplied signatures and decisions; never create a passing decision."""
    for attestation in attestations:
        stores.approval.attestation_store.persist(attestation)
    try:
        grant = stores.approval.authorization_store.load(authorization_id)
    except KeyError:
        grant = stores.approval.service.authorize(
            policy=system.authority_policy,
            frozen_proposal=identity(frozen, frozen.frozen_proposal_id),
            attestations=tuple(identity(value, value.attestation_id) for value in attestations),
            authorization_id=authorization_id, approver_id=approver_id,
        )
    if (grant.envelope.approver_id != approver_id or
        tuple(sorted(grant.envelope.attestations, key=lambda value: value.attestation_id)) !=
        tuple(sorted(attestations, key=lambda value: value.attestation_id))):
        raise ValueError("retained authorization binds another reviewer or checker decision")
    stores.approval.service.replay(
        system=system, frozen_proposal=identity(frozen, frozen.frozen_proposal_id),
        authorization=identity(grant, grant.authorization_id),
    )
    return grant
