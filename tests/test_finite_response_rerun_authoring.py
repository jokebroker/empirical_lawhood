"""Full-size software issue/package proofs; no native scientific campaign is run."""

from pathlib import Path
from time import perf_counter
from contextlib import nullcontext
import json

import pytest

from empirical_lawhood.adapters.composition.finite_response_law.assignment import proposed_scientific_seeds
from empirical_lawhood.adapters.composition.finite_response_law.rerun_input import (
    FiniteResponseLawCurrentAllocation, FiniteResponseLawCurrentExposure,
    original_development_exposure,
)
from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec
from empirical_lawhood.adapters.composition.finite_response_law.consumer_ports import FiniteResponseLawRuntimeContext
from empirical_lawhood.api import finite_rerun_authoring as application
from empirical_lawhood.api import integration_handoffs, composition as public_composition
from empirical_lawhood.api.finite_operands import current_nomination_operand_export
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind
from empirical_lawhood.runtime.artifacts import ExternalRootContract
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile, OperatorStorageAccessMode
from empirical_lawhood.api.results import CompileCampaignRequest
from tests.finite_response_rerun_fixtures import (
    publish_current_issue, publish_native_parent, publish_current_step_output,
    nominated_synthetic_native, qualify_nominated_fixture,
)
from empirical_lawhood.adapters.composition.finite_response_law.exposure import native_seed_ids
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.infrastructure.study_issue import ExternalStudyOperationAuthorityStore
from dataclasses import replace


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def finite_plane(tmp_path):
    store = tmp_path / "synthetic-finite-store"
    store.mkdir()
    return ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract(
        "synthetic.finite-store", "synthetic finite software store", str(store), "/",
        OperatorStorageProfile.SCHEMA, 1, None, None, (),
    )))


@pytest.fixture
def nomination(tmp_path):
    directory = tmp_path / "authentic-pinned-nomination"
    directory.mkdir()
    operand = current_nomination_operand_export(directory, nomination_id="synthetic.authentic-fixed-nomination")
    return operand, directory


def _allocation(stage, master_seed):
    return FiniteResponseLawCurrentAllocation(stage,
        f"empirical-lawhood.finite-response-law.{stage}.application-test-{master_seed}",
        FiniteResponseLawScienceSpec().plan_sha256, 32 if stage == "calibration" else 64,
        "PROPOSED_UNRUN", proposed_scientific_seeds(stage, master_seed))


def _software_application_seams(monkeypatch, plane):
    """Explicit fictional source, location and signed fixture clock seams only."""
    from tests.approval_support import FixedDecisionClock
    closure = ImplementationSourceClosure("synthetic.current-source", SourceClosureKind.CLEAN_GIT_COMMIT,
        "0" * 40, "d" * 64, "e" * 64, True)
    monkeypatch.setattr(application, "capture_clean_target_closure", lambda *args: closure)
    monkeypatch.setattr(public_composition, "require_executing_target_source", lambda *args: None)
    monkeypatch.setattr(public_composition, "SystemDecisionClock", lambda: FixedDecisionClock(
        clock_id="synthetic.current-prospective-evaluation.clock", value="2026-10-07T00:00:00Z"))
    monkeypatch.setattr(application, "resolve_authoring_directory", lambda directory, **kwargs: (directory, plane.root.contract))
    monkeypatch.setattr(integration_handoffs, "resolve_authoring_directory", lambda directory, **kwargs: (directory, plane.root.contract))
    monkeypatch.setattr(public_composition, "resolve_authoring_directory", lambda directory, **kwargs: (directory, plane.root.contract))
    monkeypatch.setattr(public_composition, "resolve_external_root_contract", lambda *args, **kwargs: plane.root.contract)
    monkeypatch.setattr(public_composition, "resolve_scientific_scratch_contract", lambda *args, **kwargs: plane.root.contract)


def _selected_result(plane, issue, receipt, kind):
    publication = issue.publication
    return application.current_result_input_from_receipt(config_id=f"synthetic.{kind}.selected-result",
        run_id=publication.run_id, issued_study_id=publication.issued_study.object_id,
        execution_authority_id=publication.execution_authority.object_id,
        reveal_authority_id=publication.reveal_authority.object_id, task_id=receipt.task_id,
        receipt_id=receipt.receipt_id, result_kind=kind, artifact_writer=plane)


def _print_handoff_timing(packet, handoff, started, authored, loaded, proved, capsys=None):
    """Observe already-required public calls; assert no timing or science claim."""
    with capsys.disabled() if capsys is not None else nullcontext():
        print('public_handoff_seconds', {'family': 'finite-response-law', 'stage': packet.stage,
            'config_id': packet.config_id, 'independent_roots': packet.allocation.root_count,
            'tasks': len(handoff.projection.tasks),
            'outputs': sum(len(task.outputs) for task in handoff.projection.tasks),
            'resource_cells': len(handoff.resources.task_cells),
            'author': authored-started, 'load': loaded-authored, 'prove': proved-loaded}, flush=True)


def test_current_calibration32_neutral_public_handoff_without_issued_execution(nomination, finite_plane, monkeypatch, capsys):
    """Exercise the changed dispatch/proof owners without repeating issue/recovery."""
    operand, nomination_directory = nomination
    original_units, original_seeds = original_development_exposure()
    exposure = FiniteResponseLawCurrentExposure('synthetic.neutral-proof.exposure', original_units, original_seeds)
    packet = application.prepare_finite_response_rerun(config_id='synthetic.neutral-proof.calibration32', stage='calibration',
        allocation=_allocation('calibration', 202610090032), exposure=exposure, nomination=operand,
        nomination_directory=nomination_directory, artifact_writer=finite_plane)
    _software_application_seams(monkeypatch, finite_plane)
    output = Path(finite_plane.root.contract.canonical_path) / 'neutral-calibration32'
    started = perf_counter()
    summary = application.author_finite_response_rerun(root=ROOT, packet=packet, exposure=exposure,
        nomination_directory=nomination_directory, output_dir=output, artifact_writer=finite_plane)
    authored = perf_counter()
    public = integration_handoffs.load_integration_handoff(directory=output, root=ROOT,
        storage_profile=None, artifact_writer=finite_plane)
    loaded = perf_counter()
    proof = integration_handoffs.prove_integration_handoff(public)
    handoff = public.selected
    _print_handoff_timing(packet, handoff, started, authored, loaded, perf_counter(), capsys)
    assert public.family == 'finite-response-law'
    assert summary['independent_units'] == proof['independent_units'] == len(packet.source.roots) == 32
    assert sum(task.task_id.endswith(('.flh-project.r1', '.flh-project.r2')) for task in handoff.projection.tasks) == 64
    assert summary['task_count'] == proof['task_count'] == len(handoff.projection.tasks) == len(handoff.resources.task_cells)
    assert proof['output_contract_count'] > 0
    assert all(cell.maximum_attempts == 1 and cell.jit_cell_id is None for cell in handoff.resources.task_cells)
    assert not proof['scientific_execution_performed'] and not proof['runtime_authority_bound']
    parameters = integration_handoffs.integration_api_inputs(public)
    assert parameters['executable_platform_ports'] == handoff.platform_ports
    assert parameters['study_bundle_registry'] == handoff.bundle.standard_context.base.registry


def test_full_calibration_public_authoring_and_installed_provider_resource_proof(nomination, finite_plane, tmp_path, monkeypatch, capsys):
    operand, nomination_directory = nomination
    original_units, original_seeds = original_development_exposure()
    exposure = FiniteResponseLawCurrentExposure("synthetic.original-current-exposure", original_units, original_seeds)
    packet = application.prepare_finite_response_rerun(config_id="synthetic.finite-calibration", stage="calibration",
        allocation=_allocation("calibration", 202610070032), exposure=exposure, nomination=operand,
        nomination_directory=nomination_directory, artifact_writer=finite_plane)
    _software_application_seams(monkeypatch, finite_plane)
    output = Path(finite_plane.root.contract.canonical_path) / "full-calibration-authoring"
    started = perf_counter()
    summary = application.author_finite_response_rerun(root=ROOT, packet=packet, exposure=exposure,
        nomination_directory=nomination_directory, output_dir=output, artifact_writer=finite_plane)
    authored = perf_counter()
    handoff = application.load_finite_response_rerun_handoff(directory=output, root=ROOT,
        storage_profile=None, artifact_writer=finite_plane)
    loaded = perf_counter()
    proof = application.prove_finite_response_rerun_handoff(handoff)
    _print_handoff_timing(packet, handoff, started, authored, loaded, perf_counter(), capsys)
    assert summary["independent_units"] == proof["independent_units"] == 32
    assert not summary["scientific_execution_performed"] and not summary["authority_created"]
    assert not proof["scientific_execution_performed"] and not proof["runtime_authority_bound"]
    assert summary["task_count"] == proof["task_count"] == len(handoff.resources.task_cells)
    assert proof["runner_count"] == len(handoff.bundle.standard_context.base.registry.capabilities)
    assert proof["output_contract_count"] > 0
    tasks = handoff.projection.tasks
    assert len([task for task in tasks if task.task_id.endswith((".flh-project.r1", ".flh-project.r2"))]) == 64
    assert set(packet.allocation.physical_unit_ids).isdisjoint(exposure.excluded_unit_ids)
    assert all(cell.maximum_attempts == 1 for cell in handoff.resources.task_cells)
    assert all(cell.jit_cell_id is None for cell in handoff.resources.task_cells)
    issue = publish_current_issue(finite_plane, handoff, label="current-native-calibration")
    print("current_calibration_control_bytes", dict(issue.control_sizes), flush=True)
    second_packet = application.prepare_finite_response_rerun(config_id="synthetic.second-finite-calibration", stage="calibration",
        allocation=_allocation("calibration", 202610080032), exposure=exposure, nomination=operand,
        nomination_directory=nomination_directory, artifact_writer=finite_plane)
    second_output = Path(finite_plane.root.contract.canonical_path) / "second-calibration-authoring"
    started = perf_counter()
    application.author_finite_response_rerun(root=ROOT, packet=second_packet, exposure=exposure,
        nomination_directory=nomination_directory, output_dir=second_output, artifact_writer=finite_plane)
    authored = perf_counter()
    second_handoff = application.load_finite_response_rerun_handoff(directory=second_output, root=ROOT,
        storage_profile=None, artifact_writer=finite_plane)
    loaded = perf_counter()
    second_proof = application.prove_finite_response_rerun_handoff(second_handoff)
    _print_handoff_timing(second_packet, second_handoff, started, authored, loaded, perf_counter(), capsys)
    assert second_proof["independent_units"] == 32
    assert second_handoff.projection.source_plan.object_id != handoff.projection.source_plan.object_id
    assert set(second_packet.allocation.physical_unit_ids).isdisjoint(packet.allocation.physical_unit_ids)
    assert set(native_seed_ids(second_packet.source)).isdisjoint(native_seed_ids(packet.source))
    native = nominated_synthetic_native(packet.source, operand, nomination_directory)
    parent, native, receipt, native_artifact, receipt_artifact = publish_native_parent(finite_plane, handoff, issue,
        label="current-native-calibration", native=native)
    native_result = _selected_result(finite_plane, issue, receipt, "calibration-native")
    native_summary = application.finite_response_current_result_summary(config=native_result, record=native)
    assert native_summary["independent_roots"] == 32 and native_summary["nested_numerical_views"] == 64
    assert len(native_summary["root_ids"]) == 32 and len(native_summary["views"]) == 64
    assert native_summary["result_reference"]["size_bytes"] == len(native.canonical_bytes())
    assert len(json.dumps(native_summary).encode("utf-8")) < 1024**2
    plan_path = finite_plane.root.resolve(f"runs/{receipt.run_id}/plans/execution-plan.json", for_write=False)
    primary_path = finite_plane.root.resolve(next(manifest.materialization.relative_path for manifest in native_result.manifests
        if manifest.logical.logical_artifact_id == native_result.result_artifact_id), for_write=False)
    original_plan = plan_path.read_bytes()
    actual_read = application.read_bounded_bytes
    contacted = []

    def observed_read(path, **kwargs):
        contacted.append(Path(path))
        return actual_read(path, **kwargs)

    plan_path.write_bytes(original_plan + b" ")
    try:
        with monkeypatch.context() as guarded:
            guarded.setattr(application, "read_bounded_bytes", observed_read)
            with pytest.raises((OSError, ValueError, RuntimeError)):
                application.read_finite_response_current_result(config=native_result, artifact_writer=finite_plane)
        assert primary_path not in contacted
    finally:
        plan_path.write_bytes(original_plan)
    parent = application.current_parent_from_results(parent_id=parent.parent_id, native_result=native_result,
        artifact_writer=finite_plane)
    authenticated = application.authenticate_finite_current_parent(parent, finite_plane, nomination=operand)
    assert authenticated[native_artifact.artifact_id] == native.canonical_bytes()
    protected_verifications = []
    parent_output_ids = {manifest.logical.logical_artifact_id for manifest in parent.manifests}
    original_verify = finite_plane.verify_manifests

    def observed_verify(manifests):
        if any(manifest.logical.logical_artifact_id in parent_output_ids for manifest in manifests):
            protected_verifications.append(manifests)
        return original_verify(manifests)

    with monkeypatch.context() as guarded:
        guarded.setattr(finite_plane, "verify_manifests", observed_verify)
        with pytest.raises(ValueError, match="COMPLETE_RECEIPT"):
            application.authenticate_finite_current_parent(replace(parent, manifests=parent.manifests[:-1]), finite_plane)
    assert not protected_verifications
    with pytest.raises((ValueError, PermissionError)):
        application.authenticate_finite_current_parent(replace(parent,
            execution_authority=replace(parent.execution_authority, object_fingerprint="0" * 64)), finite_plane)
    store = ExternalStudyOperationAuthorityStore(finite_plane)
    valid_execution = store.load(parent.execution_authority.object_id)
    wrong_subject = replace(valid_execution, authority_id="synthetic.wrong-subject.execution",
        subject=replace(parent.issued_study, object_id="synthetic.foreign-issued-study"))
    store.persist(wrong_subject)
    with pytest.raises((ValueError, PermissionError)):
        application.authenticate_finite_current_parent(replace(parent,
            execution_authority=ObjectIdentity.from_record(wrong_subject.authority_id, wrong_subject)), finite_plane)
    with pytest.raises((OSError, ValueError, PermissionError)):
        application.read_finite_response_current_result(config=replace(native_result,
            publication=replace(native_result.publication, run_id="synthetic.foreign-run")), artifact_writer=finite_plane)
    with pytest.raises((OSError, ValueError, PermissionError)):
        application.authenticate_finite_current_parent(replace(parent,
            receipt_bindings=(("synthetic.foreign-run", receipt.task_id, receipt.receipt_id),)), finite_plane)
    method_exposure = FiniteResponseLawCurrentExposure("synthetic.after-native-calibration-exposure",
        tuple(sorted((*original_units, *packet.allocation.physical_unit_ids))),
        tuple(sorted(set((*original_seeds, *native_seed_ids(packet.source))))))
    method_packet = application.prepare_finite_response_rerun(config_id="synthetic.finite-calibration-method",
        stage="calibration-method", allocation=packet.allocation, exposure=method_exposure,
        nomination=operand, nomination_directory=nomination_directory, artifact_writer=finite_plane, parent=parent)
    method_output = Path(finite_plane.root.contract.canonical_path) / "full-calibration-method-authoring"
    started = perf_counter()
    method_summary = application.author_finite_response_rerun(root=ROOT, packet=method_packet, exposure=method_exposure,
        nomination_directory=nomination_directory, output_dir=method_output, artifact_writer=finite_plane, parent=parent)
    authored = perf_counter()
    method_handoff = application.load_finite_response_rerun_handoff(directory=method_output, root=ROOT,
        storage_profile=None, artifact_writer=finite_plane)
    loaded = perf_counter()
    method_proof = application.prove_finite_response_rerun_handoff(method_handoff)
    _print_handoff_timing(method_packet, method_handoff, started, authored, loaded, perf_counter(), capsys)
    assert method_summary["task_count"] == method_proof["task_count"] == len(method_handoff.resources.task_cells)
    assert method_proof["independent_units"] == 32 and not method_proof["scientific_execution_performed"]
    assert not method_proof["runtime_authority_bound"]
    method_issue = publish_current_issue(finite_plane, method_handoff, label="current-calibration-method")
    qualification = qualify_nominated_fixture(finite_plane, method_packet.method, native, receipt, nomination_directory)
    qualification_receipt, _, _ = publish_current_step_output(finite_plane, method_issue,
        label="current-calibration-qualification", record=qualification)
    qualification_result = _selected_result(finite_plane, method_issue, qualification_receipt, "qualification")
    qualification_summary = application.finite_response_current_result_summary(config=qualification_result, record=qualification)
    assert qualification_summary["eligible_for_prospective_evaluation"]
    assert len(qualification_summary["boundaries"]) == 4
    for summary_boundary, scientific_boundary, qualified_boundary in zip(qualification_summary["boundaries"],
            qualification.calibration.boundaries, qualification.qualifications, strict=True):
        assert summary_boundary["boundary"] == scientific_boundary.boundary
        assert summary_boundary["assigned_root_count"] == 32
        assert summary_boundary["finite_score_count"] + summary_boundary["infinite_score_count"] == 32
        assert summary_boundary["order_index"] == scientific_boundary.order_index
        assert summary_boundary["q"] == {"decimal": str(scientific_boundary.q)}
        assert summary_boundary["scientific_status"] == qualified_boundary.scientific_status.value
    qualified_parent = application.current_parent_from_results(parent_id="synthetic.current-qualified-parent",
        native_result=native_result, qualification_result=qualification_result, artifact_writer=finite_plane)
    assert len(qualified_parent.additional_publications) == 1
    evaluation_packet = application.prepare_finite_response_rerun(config_id="synthetic.finite-prospective-evaluation",
        stage="prospective-evaluation", allocation=_allocation("prospective-evaluation", 202610070064),
        exposure=method_exposure, nomination=operand, nomination_directory=nomination_directory,
        artifact_writer=finite_plane, parent=qualified_parent)
    checkpoint = Path(finite_plane.root.contract.canonical_path) / "checkpoint-prospective-evaluation"
    checkpoint.mkdir()
    for name, record in (("packet", evaluation_packet), ("exposure", method_exposure),
            ("parent", qualified_parent), ("nomination", operand)):
        (checkpoint / f"{name}.json").write_bytes(record.canonical_bytes())
    evaluation_output = Path(finite_plane.root.contract.canonical_path) / "full-prospective-evaluation-authoring"
    finish_evaluation_public_join(finite_plane=finite_plane, evaluation_packet=evaluation_packet,
        exposure=method_exposure, nomination_directory=nomination_directory, evaluation_output=evaluation_output,
        qualified_parent=qualified_parent, tmp_path=tmp_path, foreign_issued_study=issue.publication.issued_study, capsys=capsys)


def finish_evaluation_public_join(*, finite_plane, evaluation_packet, exposure, nomination_directory,
                                  evaluation_output, qualified_parent, tmp_path, foreign_issued_study, capsys=None):
    """Complete the same public E64 joins from a retained software-only checkpoint."""
    started = perf_counter()
    evaluation_summary = application.author_finite_response_rerun(root=ROOT, packet=evaluation_packet,
        exposure=exposure, nomination_directory=nomination_directory, output_dir=evaluation_output,
        artifact_writer=finite_plane, parent=qualified_parent)
    authored = perf_counter()
    global_handoff = integration_handoffs.load_integration_handoff(directory=evaluation_output, root=ROOT,
        storage_profile=None, artifact_writer=finite_plane)
    loaded = perf_counter()
    evaluation_handoff = global_handoff.selected
    evaluation_proof = integration_handoffs.prove_integration_handoff(global_handoff)
    _print_handoff_timing(evaluation_packet, evaluation_handoff, started, authored, loaded, perf_counter(), capsys)
    assert global_handoff.family == "finite-response-law"
    parameters = integration_handoffs.integration_api_inputs(global_handoff)
    assert parameters["study_bundle_registry"] == evaluation_handoff.bundle.standard_context.base.registry
    assert parameters["executable_platform_ports"] == evaluation_handoff.platform_ports
    assert evaluation_summary["independent_units"] == evaluation_proof["independent_units"] == 64
    assert len(evaluation_packet.source.roots) == 64
    assert set(evaluation_packet.allocation.physical_unit_ids).isdisjoint(exposure.excluded_unit_ids)
    assert evaluation_proof["task_count"] == len(evaluation_handoff.resources.task_cells)
    assert all(cell.maximum_attempts == 1 and cell.jit_cell_id is None for cell in evaluation_handoff.resources.task_cells)
    assert not evaluation_proof["scientific_execution_performed"] and not evaluation_proof["runtime_authority_bound"]
    evaluation_issue = publish_current_issue(finite_plane, evaluation_handoff, label="current-prospective-evaluation")
    print("current_evaluation_control_bytes", dict(evaluation_issue.control_sizes), flush=True)
    publication = evaluation_issue.publication
    authority = ExternalStudyOperationAuthorityStore(finite_plane).load(publication.execution_authority.object_id)
    context = FiniteResponseLawRuntimeContext("synthetic.current-evaluation-runtime", publication.run_id,
        evaluation_packet.source.fingerprint(), publication.issued_study, authority.prerequisite_authority,
        publication.execution_authority, authority.grantee_id, "synthetic.software-compiler",
        publication.reveal_authority.object_id)
    for changed in (replace(context, source_sha256="0" * 64), replace(context, run_id="synthetic.foreign-run")):
        with pytest.raises(ValueError, match="PROPOSED_SOURCE_OR_RUN_MISMATCH"):
            application.bind_finite_response_runtime(directory=evaluation_output, context=changed, artifact_writer=finite_plane)
    with pytest.raises((ValueError, PermissionError)):
        application.bind_finite_response_runtime(directory=evaluation_output,
            context=replace(context, issued_study=foreign_issued_study), artifact_writer=finite_plane)
    bound = application.bind_finite_response_runtime(directory=evaluation_output, context=context, artifact_writer=finite_plane)
    assert bound == ObjectIdentity.from_record(context.context_id, context)
    runtime_global = integration_handoffs.load_integration_handoff(directory=evaluation_output, root=ROOT,
        storage_profile=None, artifact_writer=finite_plane)
    assert integration_handoffs.prove_integration_handoff(runtime_global)["runtime_authority_bound"]
    profile = OperatorStorageProfile("synthetic.finite-cli-storage", "external-filesystem", "1.0.0",
        str(tmp_path), str(tmp_path), "artifacts", "scratch", OperatorStorageAccessMode.READ_WRITE,
        None, None, (), "strict-mount-contained-no-symlink", 1, (), 1, False)
    cli_api = public_composition.create_cli_api(repo_root=ROOT, authoring_dir=evaluation_output,
        operator_storage_profile=profile, approval_checker_trust_path=evaluation_issue.approval_trust_path)
    package_path = finite_plane.root.resolve(f"runs/{publication.run_id}/plans/campaign-package.json", for_write=False)
    compiled = cli_api.compile_campaign(CompileCampaignRequest(package_path))
    assert compiled.succeeded, (compiled.reason_codes, compiled.errors)
    assert len(compiled.payload.parallel_task_groups) > 0
    verify_retained_runtime_control_recovery(cli_api, evaluation_issue)
    return evaluation_issue


def verify_retained_runtime_control_recovery(cli_api, issue):
    """Replay persisted typed plans and every static output owner without execution."""
    from empirical_lawhood.api.execution import _CatalogSemanticReplayBudget, MAX_CATALOG_REBUILD_PLAN_PACKAGE_SOURCE_BYTES

    service = cli_api._execution_service
    assert service is not None
    run_id = issue.publication.run_id
    persist_authenticated_provider_reconstruction(service, issue.package)
    assert service._load_status_execution_plan(run_id) == issue.execution_plan
    semantics = service._authenticated_rebuild_plan_semantics(run_id,
        budget=_CatalogSemanticReplayBudget(remaining_bytes=MAX_CATALOG_REBUILD_PLAN_PACKAGE_SOURCE_BYTES))
    expected = {output.logical_artifact_id: output.relative_path for task in issue.execution_plan.tasks for output in task.outputs}
    assert set(semantics) == set(expected)
    assert {key: value[0] for key, value in semantics.items()} == expected
    print("actual_evaluation_status_and_catalog_recovery_passed", len(expected), flush=True)


def persist_authenticated_provider_reconstruction(service, package):
    """Retain the real no-execution factory receipt after adjacent grant replay."""
    from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
    from empirical_lawhood.runtime.artifacts import ArtifactWriteRequest, ArtifactProfile, ArtifactLineageParent, lineage_parent_sort_key

    service._validate_campaign_authorization(package)
    service._validate_issued_execution_authority(package, at_utc=None)
    resolved = service._resolve_campaign_provider(package, authorization_replayed=True)
    reconstruction = resolved.reconstruction
    assert reconstruction is not None
    if service._load_provider_reconstruction_receipt(package.run_plan_id) is not None:
        assert service._load_provider_reconstruction_receipt(package.run_plan_id) == reconstruction
        return
    package_parents = service._package_lineage_parents(package)
    package_parent = ArtifactLineageParent(ObjectIdentity.from_record(package.package_id, package),
        VisibilityCeiling.most_restrictive(*(value.visibility_ceiling for value in package_parents)),
        service._most_restrictive_access(package_parents))
    parents = tuple(sorted((package_parent, ArtifactLineageParent(reconstruction.payload_replay_set,
        VisibilityCeiling.PROSPECTIVE, OutcomeAccess.OUTCOME_BLIND)), key=lineage_parent_sort_key))
    run_id = package.run_plan_id
    service.artifact_plane.write(ArtifactWriteRequest(logical_artifact_id=f"provider-reconstruction.{run_id}",
        relative_path=f"runs/{run_id}/plans/provider-reconstruction.json", payload_schema=reconstruction.SCHEMA,
        profile=ArtifactProfile.CANONICAL_JSON, media_type="application/json",
        publication_scope_id=f"campaign-plans.{run_id}", publication_scope_relative_root=f"runs/{run_id}/plans",
        payload=reconstruction.canonical_bytes(), logical_content_sha256=reconstruction.fingerprint(),
        visibility_ceiling=VisibilityCeiling.most_restrictive(*(value.visibility_ceiling for value in parents)),
        parent_visibility_ceilings=tuple(value.visibility_ceiling for value in parents),
        outcome_access=service._most_restrictive_access(parents), lineage_parents=parents, minimum_free_bytes=1))
    print("actual_authorized_provider_reconstruction_receipt_retained", reconstruction.receipt_id, flush=True)


def test_cancellation_only_prediction_declines_both_consumers_with_narrow_finite_uncertainty():
    """Accurate cancelling primitive forecasts cannot meet a nonzero consumer target."""
    import numpy as np
    from empirical_lawhood.adapters.methods.finite_response_law.intervals import choose

    positive = np.full((1, 5, 4, 8), 0.04)
    negative = np.full_like(positive, -0.04)
    composed = positive + negative
    sigma = np.full_like(composed, 1e-6)
    support = np.ones((1, 5), dtype=np.bool_)
    directions = np.zeros((1, 256, 2), dtype=np.int64)
    requirements = np.full(directions.shape, 0.03)
    choice = choose(composed, sigma, 1.0, support, directions, requirements)
    assert np.isfinite(choice.halfwidth).all() and (choice.halfwidth < 0.01).all()
    assert choice.selected.shape == (1, 5, 256, 2)
    assert (choice.selected == -1).all() and not choice.feasible.any()


@pytest.mark.parametrize("phase", ("parent", "future"))
def test_forecast_refuses_post_prefix_native_products_before_native_or_law_contact(phase, monkeypatch):
    from empirical_lawhood.adapters.methods.finite_response_law import control_forecast
    from empirical_lawhood.adapters.simulators.finite_response_law.contracts import native_invocations
    from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawAssignedEvaluationTaskResult
    from tests.test_finite_response_assigned_evaluation_binding import _assigned_prospective_evaluation

    _, source, control = _assigned_prospective_evaluation()
    invocation = next(value for value in native_invocations(source) if value.phase == phase)
    predecessor = ObjectIdentity(f"{invocation.predecessor_segment_id}.result",
        FiniteResponseLawAssignedEvaluationTaskResult.SCHEMA, "1.0.0", "0" * 64)
    product = FiniteResponseLawAssignedEvaluationTaskResult(invocation, (predecessor,),
        None, None, None, "PREFIX_UNAVAILABLE")

    def forbidden(*args, **kwargs):
        raise AssertionError("post-prefix product contacted native data or a law evaluator")

    monkeypatch.setattr(control_forecast, "decode_task_native", forbidden)
    monkeypatch.setattr(control_forecast, "predict_control_table", forbidden)
    with pytest.raises(ValueError, match="issued prefix or frozen qualification"):
        control_forecast.forecast_root(config=control, report=None, contexts=(), prefix=product,
            prefix_bytes=b"post-prefix bytes must stay unread", prefix_artifacts=(), payload_reader=None)


@pytest.mark.parametrize("features", (23, 25, 48))
def test_preparent_prediction_refuses_missing_or_appended_future_features_before_law_contact(features):
    from decimal import Decimal
    from empirical_lawhood.adapters.methods.finite_response_law.control_prediction import predict_control_table

    # 48 coordinates model an attempted concatenation of the causal prefix and
    # an equally sized later observation. No context or payload may be touched.
    with pytest.raises(ValueError, match="exactly 24 causal finite features"):
        predict_control_table(context=None, report=None, boundary="composed", root_id="synthetic.root",
            prefix=(Decimal(0),) * features, prefix_source=None, prefix_artifacts=(),
            parent_index=0, payload_reader=None)


def _failed_prefix_forecast(plane):
    """A typed zero-update failure, real fixed-law tables, and no acquisition."""
    from decimal import Decimal as D
    from hashlib import sha256
    import numpy as np
    from tests.finite_response_custody_fixtures import qualified_parent, artifact
    from tests.test_finite_response_assigned_evaluation_binding import _assigned_prospective_evaluation
    from empirical_lawhood.adapters.composition.finite_response_law.design import qualification_system
    from empirical_lawhood.adapters.methods.finite_response_law.control_forecast import forecast_root
    from empirical_lawhood.adapters.methods.finite_response_law.control_plan import control_law_context
    from empirical_lawhood.adapters.methods.finite_response_law.control_records import FiniteResponseLawRootForecast, FiniteResponseLawRootControlLock
    from empirical_lawhood.adapters.methods.finite_response_law.extension_bundle import METHOD_CAPABILITY
    from empirical_lawhood.adapters.simulators.finite_response_law.contracts import native_invocations
    from empirical_lawhood.adapters.simulators.finite_response_law.source import FiniteResponseLawAssignedEvaluationDelivery, FiniteResponseLawNativePhaseData, streams_for
    from empirical_lawhood.adapters.simulators.finite_response_law.native_artifact import encode_native_pair
    from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawAssignedEvaluationTaskResult
    from empirical_lawhood.infrastructure.candidate_payloads import ExternalCandidatePayloadPlane
    from empirical_lawhood.kernel.references import ExecutableReference, SafePayloadFormat
    from empirical_lawhood.kernel.evidence import VisibilityCeiling, OutcomeAccess

    report, native, _, _, _ = qualified_parent(plane)
    _, source, control = _assigned_prospective_evaluation()
    control = replace(control, qualification=artifact("synthetic.control.qualification", report.SCHEMA, report.canonical_bytes()))
    invocation = next(value for value in native_invocations(source) if value.phase == "prefix")
    phases = []
    for refinement in (1, 2):
        trace = np.empty((0, 9), dtype=np.float64)
        delivery = FiniteResponseLawAssignedEvaluationDelivery(invocation, refinement, None, None,
            0, 0, 0, (D(0), D(0)), D(0), D(0), D(0), D(0),
            sha256(trace.tobytes()).hexdigest(), "0" * 64, streams_for(invocation)[:2],
            "NUMERICAL_FAILURE", "SYNTHETIC_ZERO_UPDATE_FAILURE")
        phases.append(FiniteResponseLawNativePhaseData(delivery, None, np.array([0], dtype=np.int64),
            np.zeros((1, 2, 3, 4, 4), dtype=np.complex128), np.zeros((1, 2, 3, 4, 4), dtype=np.complex128),
            np.zeros((1, 2)), np.zeros((1, 15, 15)), np.zeros(1, dtype=np.bool_), trace))
    pair, raw = encode_native_pair(*phases)
    prefix = FiniteResponseLawAssignedEvaluationTaskResult(invocation, (), pair, None, None, None)
    prefix_artifact = artifact(prefix.result_id, prefix.SCHEMA, prefix.canonical_bytes())
    method = ExecutableReference("synthetic.finite-control.method", "synthetic.finite-control", "1.0.0",
        "synthetic.finite-control", artifact(control.config_id, control.SCHEMA, control.canonical_bytes()),
        SafePayloadFormat.CANONICAL_JSON, FiniteResponseLawRootForecast.SCHEMA,
        FiniteResponseLawRootControlLock.SCHEMA, True)
    contexts = tuple(control_law_context(system=qualification_system(native.config.projection.native_spec),
        report=report, boundary=boundary, method=method,
        producer=ObjectIdentity.from_record(METHOD_CAPABILITY.capability_key, METHOD_CAPABILITY),
        resource=ObjectIdentity.from_record("synthetic.control.resources", METHOD_CAPABILITY.resource_ceiling),
        authority=ObjectIdentity("synthetic.control.authority", "empirical-lawhood/test/authority", "1.0.0", "0" * 64))
        for boundary in ("cached", "composed", "direct"))
    reader = ExternalCandidatePayloadPlane(plane, "synthetic/laws", "synthetic.laws",
        VisibilityCeiling.PROSPECTIVE, OutcomeAccess.EVALUATOR_REVEAL, (), 1)
    forecast = forecast_root(config=control, report=report, contexts=contexts, prefix=prefix,
        prefix_bytes=raw, prefix_artifacts=(prefix_artifact,), payload_reader=reader)
    assert forecast.primary_interface is None and len(forecast.tables) == 3
    return forecast


def test_partial_freeze_cannot_publish_all_consumer_guard(finite_plane):
    from tests.finite_response_custody_fixtures import artifact
    from empirical_lawhood.adapters.methods.finite_response_law.consumer import evaluation_requests
    from empirical_lawhood.adapters.methods.finite_response_law.control_records import FiniteResponseLawRootControlLock, FiniteResponseLawRootRequestReveal

    forecast = _failed_prefix_forecast(finite_plane)
    reveal = FiniteResponseLawRootRequestReveal(forecast.root,
        ObjectIdentity.from_record(forecast.forecast_id, forecast),
        artifact(forecast.forecast_id, forecast.SCHEMA, forecast.canonical_bytes()),
        "synthetic.full-forecast.receipt", evaluation_requests(forecast.root))
    # All three qualified boundaries and both consumers must have durable locks;
    # interruption before their publication cannot authorize the parent effect.
    with pytest.raises(ValueError, match="lose a qualified method/consumer"):
        FiniteResponseLawRootControlLock(forecast, reveal, (), (), ())


@pytest.mark.parametrize("refinement", (1, 2))
def test_unavailable_observed_handoff_terminates_frozen_action_and_refuses_replacement(refinement, monkeypatch):
    from empirical_lawhood.adapters.methods.finite_response_law import control_prediction, intervals
    from empirical_lawhood.adapters.methods.finite_response_law.control_delivery import FiniteResponseLawNativeReceiptDelivery, observation_words, observe_native_word
    from empirical_lawhood.adapters.methods.finite_response_law.control_services import implementation_payloads
    from empirical_lawhood.adapters.methods.finite_response_law.law_binding import NATIVE_WORDS
    from empirical_lawhood.adapters.simulators.finite_response_law import source as native_source
    from empirical_lawhood.adapters.simulators.finite_response_law.contracts import native_invocations
    from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawAssignedEvaluationTaskResult, unentered_native_bytes
    from empirical_lawhood.kernel.control import OperationalDeliveryState, ScientificCommitmentKind
    from empirical_lawhood.planning.controller_study import ImplementationRole
    from tests.test_finite_response_assigned_evaluation_binding import _assigned_prospective_evaluation

    _, source, _ = _assigned_prospective_evaluation()
    root = source.roots[0]
    original_word, replacement_word = observation_words(root.stage_unit)[:2]
    original_identity = ObjectIdentity.from_record(original_word.word_id, original_word)
    futures = tuple(value for value in native_invocations(source)
        if value.root == root and value.phase == "future" and value.word == NATIVE_WORDS[0])
    assert tuple(value.purpose for value in futures) == ("future-1", "future-2")
    inputs = tuple((FiniteResponseLawAssignedEvaluationTaskResult(invocation,
        (ObjectIdentity(f"{invocation.predecessor_segment_id}.result",
            FiniteResponseLawAssignedEvaluationTaskResult.SCHEMA, "1.0.0", "0" * 64),),
        None, None, None, "PREFIX_UNAVAILABLE"), unentered_native_bytes()) for invocation in futures)

    def forbidden(*args, **kwargs):
        raise AssertionError("cancellation contacted prediction, selection or native acquisition")

    monkeypatch.setattr(control_prediction, "predict_control_table", forbidden)
    monkeypatch.setattr(intervals, "choose", forbidden)
    monkeypatch.setattr(native_source, "execute_native_phase", forbidden)
    observed = observe_native_word(word=original_word, refinement=refinement, inputs=inputs)
    binding = next(binding for binding, _ in implementation_payloads() if binding.role is ImplementationRole.DELIVERY)
    port = FiniteResponseLawNativeReceiptDelivery(binding, observed)
    stopped = port.deliver(action_word=original_word, commitment_kind=ScientificCommitmentKind.ACTION)
    assert stopped.failure_state is OperationalDeliveryState.TERMINATED
    assert stopped.observed_occurrences == (observed.observed,) and not observed.observed.complete
    with pytest.raises(ValueError, match="substitutes the precommitted native action"):
        port.deliver(action_word=replacement_word, commitment_kind=ScientificCommitmentKind.ACTION)
    assert ObjectIdentity.from_record(port.evidence.word.word_id, port.evidence.word) == original_identity
