"""No-contact production proof for one strict fresh reactor authoring packet.

SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

from pathlib import Path

from empirical_lawhood.api.composition import create_cli_api
from empirical_lawhood.api.results import CheckReadinessRequest
from empirical_lawhood.infrastructure.artifacts import (
    ExternalArtifactPlane,
    GuardedExternalRoot,
)
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure
from empirical_lawhood.runtime.artifacts import (
    ArtifactProfile,
    ArtifactStreamWriteRequest,
    ArtifactWriteRequest,
)
from empirical_lawhood.runtime.execution_envelope import ExecutionResourceEnvelopeSpec
from empirical_lawhood.runtime.operator_profile import (
    OperatorStorageProfile,
    resolve_external_root_contract,
)
from empirical_lawhood.runtime.plans import CandidateExecutionPlan, CandidateRunPlan
from empirical_lawhood.runtime.study_issue import MAX_ISSUE_PAYLOAD_BYTES

from empirical_lawhood.adapters.composition.reactor_prefix_response.fresh_authoring import load_fresh_reactor_bundle

_RECORD_LIMIT = 16 * 1024 * 1024


def _load(
    directory: Path, filename: str, record_type: type[CanonicalRecord]
) -> CanonicalRecord:
    return decode_canonical_bytes(
        read_bounded_bytes(directory / filename, maximum_bytes=_RECORD_LIMIT),
        record_type,
        maximum_bytes=_RECORD_LIMIT,
    )


def prove_fresh_reactor(
    *,
    repo_root: Path,
    storage_profile: OperatorStorageProfile,
    directory: Path,
    approval_checker_trust_path: Path | None = None,
) -> dict[str, object]:
    """Resolve the actual CLI API and persist exact control records; run no task."""

    directory = directory.resolve(strict=True)
    issue_member_bytes = {
        name: (directory / name).stat().st_size
        for name in ("base-authoring.json", "authoring.json")
    }
    if any(size > MAX_ISSUE_PAYLOAD_BYTES for size in issue_member_bytes.values()):
        raise ValueError("REACTOR_AUTHORING_EXCEEDS_PRODUCTION_ISSUE_BYTE_BOUND")
    bundle = load_fresh_reactor_bundle(directory)
    from empirical_lawhood.adapters.composition.reactor_prefix_response.fresh_authoring import fresh_assignment_status, load_fresh_reactor_profile

    profile = load_fresh_reactor_profile(directory / "profile.json")
    run_id = bundle.authoring.base.draft.experiment.experiment_id
    root_contract = resolve_external_root_contract(
        storage_profile, repo_root=repo_root, home_root=Path.home()
    )
    guard = GuardedExternalRoot(root_contract)
    guard.verify(for_write=True)
    if not directory.is_relative_to(Path(root_contract.canonical_path)):
        raise ValueError("fresh authoring directory is outside guarded artifacts")
    api = create_cli_api(
        repo_root=repo_root,
        operator_storage_profile=storage_profile,
        approval_checker_trust_path=approval_checker_trust_path,
        reactor_authoring_dir=directory,
    )
    payloads = tuple(sorted(directory.glob("payload-*.json")))
    decoders = tuple(sorted(directory.glob("decoder-*.json")))
    result = api.preissue_readiness(
        CheckReadinessRequest(
            authoring_package_path=directory / "authoring.json",
            extension_payload_paths=payloads,
            decoder_registration_paths=decoders,
            expected_candidate_path=directory / "candidate.json",
            source_closure_path=directory / "source-closure.json",
            execution_resource_envelope_spec_path=directory / "resources.json",
            predevelopment_jit_signature_census_path=None,
            jit_graph_signature_manifest_path=None,
            run_plan_id=run_id,
        )
    )
    if not result.succeeded or result.payload is None:
        raise RuntimeError(
            f"fresh reactor preissue route refused: {result.reason_codes}; {result.errors}"
        )
    summary = result.payload
    if not all(passed for _, passed, _ in summary.closure_sections):
        raise ValueError("fresh reactor shared preissue closure did not pass")
    run_plan = _load(directory, "preissue-run-plan.json", CandidateRunPlan)
    execution = _load(directory, "preissue-execution-plan.json", CandidateExecutionPlan)
    resources = _load(directory, "resources.json", ExecutionResourceEnvelopeSpec)
    closure = _load(directory, "source-closure.json", ImplementationSourceClosure)
    assert isinstance(run_plan, CandidateRunPlan)
    assert isinstance(execution, CandidateExecutionPlan)
    assert isinstance(resources, ExecutionResourceEnvelopeSpec)
    assert isinstance(closure, ImplementationSourceClosure)
    paths = tuple(
        sorted(
            output.relative_path for task in execution.tasks for output in task.outputs
        )
    )
    if paths != summary.output_relative_locators:
        raise ValueError(
            "full preissue output locator set differs from the CLI API proof"
        )
    plane = ExternalArtifactPlane(guard)
    for task in execution.tasks:
        budget = task.capability.requested_resources.output_bytes
        plane._preflight_publication(
            tuple(
                ArtifactStreamWriteRequest(
                    logical_artifact_id=output.logical_artifact_id,
                    relative_path=output.relative_path,
                    payload_schema=output.payload_schema,
                    profile=output.profile,
                    media_type=output.media_type,
                    publication_scope_id=f"task-output-scope.{run_id}.{task.task_id}",
                    publication_scope_relative_root=f"runs/{run_id}",
                    chunks=(),
                    maximum_bytes=budget,
                    maximum_chunk_bytes=min(budget, 1024 * 1024),
                    visibility_ceiling=output.visibility_ceiling,
                    parent_visibility_ceilings=output.parent_visibility_ceilings,
                    outcome_access=output.outcome_access,
                    read_only_source_input=True,
                    minimum_free_bytes=root_contract.minimum_free_bytes,
                )
                for output in task.outputs
            )
        )
    proof_root = f"control-persistence/{run_id}/{closure.implementation_commit[:12]}"
    persisted = []
    for name, record in (
        ("source-closure", closure),
        ("resources", resources),
        ("preissue-run-plan", run_plan),
        ("preissue-execution-plan", execution),
    ):
        relative = f"{proof_root}/{name}.json"
        request = ArtifactWriteRequest(
            logical_artifact_id=f"{run_id}.{closure.implementation_commit[:12]}.control-proof.{name}",
            relative_path=relative,
            payload_schema=record.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON,
            media_type="application/vnd.empirical-lawhood.canonical+json",
            publication_scope_id=f"control-proof.{run_id}",
            publication_scope_relative_root=proof_root,
            payload=record.canonical_bytes(),
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            parent_visibility_ceilings=(),
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            minimum_free_bytes=root_contract.minimum_free_bytes,
        )
        first = plane.write_stream(request.as_stream())
        replay = plane.write_stream(request.as_stream())
        path = guard.resolve(relative, for_write=False)
        manifest = decode_artifact_manifest(
            read_bounded_bytes(
                guard.resolve(
                    first.manifest_materialization.relative_path, for_write=False
                ),
                maximum_bytes=_RECORD_LIMIT,
            )
        )
        plane.verify_manifest(manifest)
        if (
            replay.created
            or replay.materialization != first.materialization
            or path.read_bytes() != record.canonical_bytes()
        ):
            raise ValueError("preissue control record did not replay immutably")
        persisted.append(
            {
                "name": name,
                "sha256": record.fingerprint(),
                "size_bytes": len(record.canonical_bytes()),
                "relative_path": relative,
                "manifest_sha256": manifest.fingerprint(),
            }
        )
    return {
        "experiment_id": run_id,
        "implementation_commit": closure.implementation_commit,
        "candidate_sha256": summary.candidate_fingerprint,
        "task_count": summary.task_count,
        "runner_ids": summary.runner_ids,
        "external_input_ids": summary.external_input_ids,
        "output_locators": paths,
        "adjudication_locator": summary.adjudication_relative_locator,
        "aggregate_task_output_byte_ceiling": sum(
            task.capability.requested_resources.output_bytes for task in execution.tasks
        ),
        "resource_envelope_sha256": resources.fingerprint(),
        "authority_gates": summary.authority_input_ids,
        "closure_sections": summary.closure_sections,
        "control_persistence": persisted,
        "issue_authoring_member_bytes": issue_member_bytes,
        "issue_authoring_member_byte_ceiling": MAX_ISSUE_PAYLOAD_BYTES,
        **fresh_assignment_status(profile),
        "source_contacted": False,
        "native_tasks_executed": 0,
    }
