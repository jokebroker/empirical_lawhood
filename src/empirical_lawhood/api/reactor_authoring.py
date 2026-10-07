"""Application-owned candidate compilation and plan projection.

SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations
from pathlib import Path
from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY,
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.composition.reactor_prefix_response.authoring import ReactorPrefixResponseCandidateContextProvider, build_reactor_authoring
from empirical_lawhood.adapters.composition.reactor_prefix_response.resources import reactor_resource_envelope
from empirical_lawhood.adapters.simulators.reactor_prefix_response.panel import ReactorSourceBundle
from empirical_lawhood.api.facade import EmpiricalLawhoodApi
from empirical_lawhood.api.results import CompileCandidateRequest
from empirical_lawhood.adapters.simulators.reactor_prefix_response.packaged_source import load_packaged_reactor_source
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.compiler import compile_preissue_run_plan, lower_run_plan
from empirical_lawhood.runtime.study_issue import MAX_ISSUE_PAYLOAD_BYTES, project_standard_study_extensions
from empirical_lawhood.adapters.composition.reactor_prefix_response.fresh_authoring import FreshReactorAuthoringProfile, fresh_assignment_status
from empirical_lawhood.infrastructure.source_closure import capture_clean_target_closure


def _save_new(directory: Path, name: str, record: CanonicalRecord) -> Path:
    path = directory / name
    with path.open("xb") as handle:
        handle.write(record.canonical_bytes())
        handle.flush()
    return path


def author_fresh_reactor(
    *, root: Path, profile: FreshReactorAuthoringProfile, output_dir: Path
) -> dict[str, object]:
    """Create strict inputs and pure preissue plan projections; no native task runs."""

    closure = capture_clean_target_closure(
        root, f"{profile.experiment_id}.source-closure"
    )
    source: ReactorSourceBundle = load_packaged_reactor_source()
    if source.fingerprint() != profile.public_source_sha256:
        raise ValueError(
            "packaged public reactor input changed from the strict profile"
        )
    bundle = build_reactor_authoring(
        source=source,
        implementation_sha256=closure.implementation_sha256,
        fresh_experiment_id=profile.experiment_id,
        assignment=getattr(profile, "assignment", None),
        prior_census=getattr(profile, "prior_census", None),
    )
    if any(
        len(record.canonical_bytes()) > MAX_ISSUE_PAYLOAD_BYTES
        for record in (bundle.authoring.base, bundle.authoring)
    ):
        raise ValueError("REACTOR_AUTHORING_EXCEEDS_PRODUCTION_ISSUE_BYTE_BOUND")
    if output_dir.is_symlink() or output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "fresh reactor authoring output must be a new empty directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_new(output_dir, "profile.json", profile)
    authoring = _save_new(output_dir, "authoring.json", bundle.authoring)
    _save_new(output_dir, "base-authoring.json", bundle.authoring.base)
    payloads = tuple(
        _save_new(output_dir, f"payload-{index}.json", record)
        for index, record in enumerate(bundle.payloads)
    )
    decoders = tuple(
        _save_new(output_dir, f"decoder-{index}.json", record)
        for index, record in enumerate(bundle.decoder_registrations)
    )
    api = EmpiricalLawhoodApi(
        repo_root=root,
        external_root=None,
        candidate_context_provider=ReactorPrefixResponseCandidateContextProvider(bundle),
        candidate_capability_catalog=bundle.catalog,
        extension_bundle_aggregate=GENERATED_EXTENSION_BUNDLE_AGGREGATE,
        executable_binding_aggregate=EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.aggregate,
        executable_factory_registry=EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
        study_extension_codec_registry=EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY,
    )
    compiled = api.compile_candidate(
        CompileCandidateRequest(authoring, payloads, decoders)
    )
    if not compiled.succeeded or compiled.payload is None:
        raise RuntimeError(f"fresh reactor candidate refused: {compiled.reason_codes}")
    report = compiled.payload.report
    candidate = report.candidate
    if candidate is None:
        raise RuntimeError("fresh reactor candidate compiler returned no candidate")
    base = candidate.base_candidate.base_candidate
    issued_extensions, _, _ = project_standard_study_extensions(
        candidate=candidate,
        extension_payload_bytes=tuple(
            record.canonical_bytes() for record in bundle.payloads
        ),
        decoder_registrations=bundle.decoder_registrations,
        extension_materializations=report.extension_materializations,
    )
    resources = reactor_resource_envelope(
        base.protocol,
        ObjectIdentity.from_record(
            issued_extensions.issued_extension_set_id, issued_extensions
        ),
        run_id=profile.experiment_id,
        native_config=next(p for p in bundle.payloads if hasattr(p, "branches")),
    )
    projected = compile_preissue_run_plan(
        run_plan_id=profile.experiment_id,
        candidate_record=base,
        candidate=ObjectIdentity.from_record(candidate.candidate_id, candidate),
        registry=bundle.standard_context.base.registry,
        implementation_commit=closure.implementation_commit,
        issued_extension_set=issued_extensions,
        resource_envelope=resources,
        jit_census=None,
        jit_manifest=None,
    )
    lowered = lower_run_plan(projected, bundle.standard_context.base.registry)
    _save_new(output_dir, "candidate-report.json", report)
    _save_new(output_dir, "candidate.json", candidate)
    _save_new(output_dir, "base-candidate.json", candidate.base_candidate)
    _save_new(output_dir, "source-closure.json", closure)
    _save_new(output_dir, "resources.json", resources)
    _save_new(output_dir, "preissue-run-plan.json", projected)
    _save_new(output_dir, "preissue-execution-plan.json", lowered)
    _save_new(output_dir, "evidence-profile.json", bundle.evidence_profile)
    return {
        "experiment_id": profile.experiment_id,
        "implementation_commit": closure.implementation_commit,
        "candidate_sha256": candidate.fingerprint(),
        "task_count": len(lowered.tasks),
        "output_count": sum(len(task.outputs) for task in lowered.tasks),
        "authoring_dir": str(output_dir),
        **fresh_assignment_status(profile),
        "source_contacted": False,
        "native_tasks_executed": 0,
    }
