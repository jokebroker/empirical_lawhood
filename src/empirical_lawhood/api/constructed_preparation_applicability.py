"""Public strict current authoring and retained handoff for one fixed family."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from empirical_lawhood.adapters.composition.constructed_preparation_applicability.selection import ConstructedPreparationApplicabilitySelection
from empirical_lawhood.adapters.composition.constructed_preparation_applicability.authoring import build_authoring, SPECIFICATION_PATH
from empirical_lawhood.adapters.composition.constructed_preparation_applicability.resources import resource_envelope
from empirical_lawhood.adapters.composition.phase_authoring import PhaseAuthoringBundle, PhaseContextProvider
from empirical_lawhood.adapters.composition.preparation_applicability.ports import runtime_platform_ports
from empirical_lawhood.api.authoring_handoff import create_authoring_api, write_exclusive_record, resolve_authoring_directory, load_authoring_execution_projection
from empirical_lawhood.api.authoring_output import report_incomplete_output
from empirical_lawhood.api.results import CompileCandidateRequest
from empirical_lawhood.adapters.methods.preparation_applicability.config import preparation_run_id
from empirical_lawhood.api.preparation_result_access import authenticate_preparation_result_access
from empirical_lawhood.api.preparation_applicability import preflight_preparation_authoring_output
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore
from empirical_lawhood.infrastructure.source_closure import capture_clean_target_closure
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure
from empirical_lawhood.runtime.artifacts import ArtifactWriter
from empirical_lawhood.runtime.compiler import compile_preissue_run_plan, lower_run_plan
from empirical_lawhood.runtime.execution_envelope import ExecutionResourceEnvelopeSpec
from empirical_lawhood.runtime.executable_bindings import ExecutablePlatformPort
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile
from empirical_lawhood.runtime.plans import CandidateExecutionPlan
from empirical_lawhood.runtime.study_issue import project_standard_study_extensions


def _bundle(selection, root, closure):
    from empirical_lawhood.adapters.methods.preparation_applicability.conformance import current_native_implementation
    if selection.source.native_implementation != current_native_implementation():
        raise ValueError("selected native implementation differs from the installed current owners")
    specification = read_bounded_bytes(root / SPECIFICATION_PATH, maximum_bytes=2 * 1024**2)
    lock = read_bounded_bytes(root / "uv.lock", maximum_bytes=4 * 1024**2)
    if (sha256(specification).hexdigest() != selection.source.implementation_plan_sha256
        or sha256(lock).hexdigest() != selection.source.dependency_lock_sha256):
        raise ValueError("selected current specification or locked dependency bytes drifted")
    return build_authoring(selection.stage, selection.source,
        implementation_sha256=closure.implementation_sha256,
        specification_sha256=sha256(specification).hexdigest(), qualified_report=selection.qualified_report)


def _authenticate(selection, writer, plane):
    for value in selection.upstream_records:
        value.authenticate(writer, ExternalTaskReceiptStore(plane), result_access_authenticator=authenticate_preparation_result_access)
        value.require_stage(selection.stage)


def author_constructed_preparation_applicability(*, root: Path, selection: ConstructedPreparationApplicabilitySelection, output_dir: Path,
                  artifact_writer: ArtifactWriter) -> dict[str, object]:
    """Export a complete candidate and preissue plan without simulation or issue."""
    output_dir=preflight_preparation_authoring_output(output_dir=output_dir,root=root,artifact_writer=artifact_writer)
    _authenticate(selection, artifact_writer, artifact_writer)
    closure = capture_clean_target_closure(root, f"{selection.stage.config_id}.source-closure")
    bundle = _bundle(selection, root, closure)
    output_dir=preflight_preparation_authoring_output(output_dir=output_dir,root=root,artifact_writer=artifact_writer)
    output_dir.mkdir(parents=True, exist_ok=False)
    with report_incomplete_output(output_dir) as progress:
        def save(name, record):
            return write_exclusive_record(output_dir, name, record)
        save("selection.json", selection)
        save("stage.json", selection.stage)
        save("source.json", selection.source)
        save("source-closure.json", closure)
        authoring = save("authoring.json", bundle.authoring)
        save("base-authoring.json", bundle.authoring.base)
        payloads = tuple(save(f"payload-{index:02d}.json", value) for index, value in enumerate(bundle.payloads))
        decoders = tuple(save(f"decoder-{index:02d}.json", value) for index, value in enumerate(bundle.decoder_registrations))
        api = create_authoring_api(repo_root=root, candidate_context_provider=PhaseContextProvider(bundle),
                                   candidate_capability_catalog=bundle.catalog)
        progress.stage = "candidate compilation"
        compiled = api.compile_candidate(CompileCandidateRequest(authoring, payloads, decoders))
        if not compiled.succeeded or compiled.payload is None or compiled.payload.report.candidate is None:
            raise RuntimeError(f"applicability candidate refused: {compiled.reason_codes}; {compiled.errors}")
        report = compiled.payload.report
        candidate = report.candidate
        extensions, _, _ = project_standard_study_extensions(candidate=candidate,
            extension_payload_bytes=tuple(value.canonical_bytes() for value in bundle.payloads),
            decoder_registrations=bundle.decoder_registrations, extension_materializations=report.extension_materializations)
        base = candidate.base_candidate.base_candidate
        resources = resource_envelope(selection.stage, base.protocol,
            ObjectIdentity.from_record(extensions.issued_extension_set_id, extensions))
        progress.stage = "plan projection"
        projected = compile_preissue_run_plan(run_plan_id=preparation_run_id(selection.stage), candidate_record=base,
            candidate=ObjectIdentity.from_record(candidate.candidate_id, candidate), registry=bundle.standard_context.base.registry,
            implementation_commit=closure.implementation_commit, issued_extension_set=extensions,
            resource_envelope=resources, jit_census=None, jit_manifest=None)
        projection = lower_run_plan(projected, bundle.standard_context.base.registry)
        for name, record in (("candidate-report.json", report), ("candidate.json", candidate),
            ("base-candidate.json", candidate.base_candidate), ("resources.json", resources),
            ("preissue-run-plan.json", projected), ("preissue-execution-plan.json", projection),
            ("evidence-profile.json", bundle.evidence_profile)):
            save(name, record)
    return {"phase": selection.stage.phase, "authoring_dir": str(output_dir),
        "candidate_sha256": candidate.fingerprint(), "task_count": len(projection.tasks),
        "independent_units": len(selection.stage.root_ids), "source_contacted": False,
        "native_tasks_executed": 0}


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityApplicationHandoff:
    directory: Path
    bundle: PhaseAuthoringBundle
    projection: CandidateExecutionPlan
    resources: ExecutionResourceEnvelopeSpec
    platform_ports: tuple[ExecutablePlatformPort, ...]


def load_constructed_preparation_applicability_handoff(*, directory: Path, root: Path, storage_profile: OperatorStorageProfile,
               artifact_writer: ArtifactWriter) -> PreparationApplicabilityApplicationHandoff:
    directory, contract = resolve_authoring_directory(directory, repo_root=root, storage_profile=storage_profile,
        escape_message="applicability authoring handoff is outside its selected guarded external root")
    def read(name, kind):
        return decode_canonical_bytes(read_bounded_bytes(directory / name, maximum_bytes=16 * 1024**2),
                                      kind, maximum_bytes=16 * 1024**2)
    selection = read("selection.json", ConstructedPreparationApplicabilitySelection)
    plane = ExternalArtifactPlane(GuardedExternalRoot(contract))
    _authenticate(selection, artifact_writer, plane)
    closure = read("source-closure.json", ImplementationSourceClosure)
    current_closure = capture_clean_target_closure(root, closure.source_closure_id)
    if current_closure != closure:
        raise ValueError("retained authoring current implementation closure differs")
    bundle = _bundle(selection, root, closure)
    if bundle.authoring != read("authoring.json", ExecutableStudyDefinition):
        raise ValueError("retained authoring differs from the exact installed family reconstruction")
    projection = load_authoring_execution_projection(directory)
    resources = read("resources.json", ExecutionResourceEnvelopeSpec)
    if projection.registry_sha256 != bundle.standard_context.base.registry.fingerprint():
        raise ValueError("retained applicability registry differs from its current family")
    from empirical_lawhood.api.preparation_applicability import _verify_handoff_projection
    _verify_handoff_projection(directory=directory,root=root,selection=selection,bundle=bundle,
        closure=closure,projection=projection,resources=resources,resource_builder=resource_envelope)
    return PreparationApplicabilityApplicationHandoff(directory, bundle, projection, resources,
        runtime_platform_ports(selection=selection, bundle=bundle, projection=projection, root=plane.root))


def prove_constructed_preparation_applicability_handoff(handoff):
    from empirical_lawhood.api.preparation_applicability import prove_preparation_applicability_handoff
    return prove_preparation_applicability_handoff(handoff)
