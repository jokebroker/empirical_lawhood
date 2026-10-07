# SPDX-License-Identifier: MPL-2.0
"""Researcher entry points for current paper integrations.

Typed/configuration and scientific owners remain in API/adapters; this module
selects explicit paths, records attempts and presents bounded operation results.
"""

from pathlib import Path
from typing import Annotated

import typer

from empirical_lawhood.cli.attempts import ATTEMPT_DIRECTORY_OPTION, ATTEMPT_OUTPUT_OPTION, attempt_input, attempt_stage, emit_development_report, retained_command



def _report(value) -> None:
    if hasattr(value, 'to_document'):
        value = value.to_document()
    emit_development_report(value)


def _error(error) -> None:
    typer.echo(str(error), err=True)
    raise typer.Exit(code=2) from error


def _selected_input(path: Path, *, maximum_bytes: int = 16 * 1024**2, copy: bool = True, consumer: str | None = None) -> Path:
    return attempt_input(path, maximum_bytes=maximum_bytes, copy=copy,
                         role="config" if consumer is None else consumer, consumer=consumer)


def _configuration(path: Path, consumer: str):
    from empirical_lawhood.api.configuration import load_configuration_record
    from empirical_lawhood.api.configuration_registry import consumer_by_id
    owner = consumer_by_id(consumer)
    return load_configuration_record(_selected_input(path, maximum_bytes=owner.authoring_maximum_bytes or 16 * 1024**2, consumer=consumer), consumer=consumer)


def _storage():
    from empirical_lawhood.cli import platform
    from empirical_lawhood.api.codecs import load_registered_authoring
    from empirical_lawhood.api.integration_handoffs import integration_artifact_profile_validators
    from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
    from empirical_lawhood.infrastructure.source_origin import require_executing_target_source
    from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile, resolve_external_root_contract
    root, profile_path = platform.CLI_PROJECT_ROOT, platform.CLI_OPERATOR_PROFILE
    if root is None or profile_path is None:
        raise ValueError('This operation requires --project-root and --operator-profile; select the storage profile described in the guide.')
    require_executing_target_source(root)
    profile = load_registered_authoring(profile_path, root_schemas={OperatorStorageProfile.SCHEMA: OperatorStorageProfile})
    plane = ExternalArtifactPlane(GuardedExternalRoot(resolve_external_root_contract(profile, repo_root=root, home_root=Path.home())), integration_artifact_profile_validators())
    plane.root.verify(for_write=False)
    return root, profile, plane


def _destination(path: Path, plane=None) -> Path:
    """Explicit new destination; no home/output fallback or overwrite."""
    if '..' in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('Output refuses symbolic links and parent traversal')
    selected = path.absolute()
    if plane is not None:
        if not selected.is_relative_to(Path(plane.root.contract.canonical_path)):
            raise ValueError('Output must be under the selected guarded external root')
        plane.root.verify(for_write=True)
    if selected.exists() and (not selected.is_dir() or tuple(selected.iterdir())):
        raise FileExistsError('Output must be a new empty directory')
    selected.mkdir(parents=True, exist_ok=True)
    return selected


def register_integration_commands(campaign_app: typer.Typer) -> None:
    from empirical_lawhood.cli.preparation_integrations import register_preparation_commands
    register_preparation_commands(campaign_app)
    from empirical_lawhood.cli.preparation_diagnostics import register_preparation_diagnostic_commands
    register_preparation_diagnostic_commands(campaign_app)
    from empirical_lawhood.cli.paper_analysis_integrations import register_paper_analysis_commands
    register_paper_analysis_commands(campaign_app)

    @campaign_app.command('finite-response-allocation')
    @retained_command('campaign finite-response-allocation', closed_output=True, native=False)
    def finite_response_allocation(
        stage: Annotated[str, typer.Option('--stage', help='calibration or prospective-evaluation')],
        cohort_namespace: Annotated[str, typer.Option('--cohort-namespace')],
        master_seed: Annotated[int, typer.Option('--master-seed')],
        output: Annotated[Path, typer.Option('--output')],
        evidence_role: Annotated[str, typer.Option('--evidence-role')] = 'PROPOSED_UNRUN',
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Expand an explicit master seed into the owned32/64-root numerical purpose roster."""
        from empirical_lawhood.api.configuration import _write_new_file, preflight_new_configuration_output
        from empirical_lawhood.api.finite_rerun_authoring import prepare_finite_response_allocation
        try:
            output = preflight_new_configuration_output(output)
            attempt_stage('explicit_current_allocation_preparation', native_contact='none')
            record = prepare_finite_response_allocation(stage=stage, cohort_namespace=cohort_namespace, master_seed=master_seed, evidence_role=evidence_role)
            _write_new_file(output, record.canonical_bytes())
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report({'status': 'PROPOSED', 'output': str(output), 'allocation_sha256': record.fingerprint(), 'evidence_role': record.evidence_role, 'scientific_freshness_established': False})

    @campaign_app.command('matrix-source-config')
    @retained_command('campaign matrix-source-config', closed_output=True, native=False)
    def matrix_source_config(
        config_id: Annotated[str, typer.Option('--config-id')],
        allocation: Annotated[Path, typer.Option('--allocation', exists=True, dir_okay=False)],
        output: Annotated[Path, typer.Option('--output')],
        original_f: Annotated[Path | None, typer.Option('--original-f', exists=True, dir_okay=False)] = None,
        source_directory: Annotated[Path | None, typer.Option('--source-directory', exists=True, file_okay=False)] = None,
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Bind edited allocations to current source code; optional F selects the fixed24 route."""
        from empirical_lawhood.api.configuration import _write_new_file, preflight_new_configuration_output
        from empirical_lawhood.api.finite_operands import original_f_operand_import, prepare_matrix_source_configuration
        try:
            output = preflight_new_configuration_output(output)
            if (original_f is None) != (source_directory is None):
                raise ValueError('Fixed24 source configuration requires original F and its exact source directory together')
            selected = _configuration(allocation, 'matrix-allocation')
            lower = None if original_f is None else original_f_operand_import(_selected_input(original_f), source_directory=source_directory)
            attempt_stage('current_source_configuration_preparation', native_contact='none')
            from empirical_lawhood.cli import platform
            record = prepare_matrix_source_configuration(config_id=config_id, allocation=selected, original_f=lower, project_root=platform.CLI_PROJECT_ROOT)
            _write_new_file(output, record.canonical_bytes())
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report({'status': 'PREPARED', 'output': str(output), 'source_sha256': record.fingerprint(), 'native_contact': 'none'})

    @campaign_app.command('finite-response-packet')
    @retained_command('campaign finite-response-packet', closed_output=True, native=False)
    def finite_response_packet(
        config_id: Annotated[str, typer.Option('--config-id')],
        stage: Annotated[str, typer.Option('--stage', help='calibration, calibration-method or prospective-evaluation')],
        allocation: Annotated[Path, typer.Option('--allocation', exists=True, dir_okay=False)],
        exposure: Annotated[Path, typer.Option('--exposure', exists=True, dir_okay=False)],
        nomination: Annotated[Path, typer.Option('--nomination', exists=True, dir_okay=False)],
        nomination_directory: Annotated[Path, typer.Option('--nomination-directory', exists=True, file_okay=False)],
        output: Annotated[Path, typer.Option('--output')],
        parent: Annotated[Path | None, typer.Option('--parent', exists=True, dir_okay=False)] = None,
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Prepare a stage packet from edited allocations and authenticated current operands."""
        from empirical_lawhood.api.configuration import _write_new_file, preflight_new_configuration_output
        from empirical_lawhood.api.finite_rerun_authoring import prepare_finite_response_rerun
        try:
            _, _, plane = _storage()
            output = preflight_new_configuration_output(output)
            selected = _configuration(allocation, 'finite-current-allocation')
            census = _configuration(exposure, 'finite-current-exposure')
            coefficients = _configuration(nomination, 'current-nomination')
            current_parent = None if parent is None else _configuration(parent, 'finite-current-parent')
            attempt_stage('current_stage_packet_preparation', native_contact='none')
            record = prepare_finite_response_rerun(config_id=config_id, stage=stage, allocation=selected, exposure=census, nomination=coefficients, nomination_directory=nomination_directory, artifact_writer=plane, parent=current_parent)
            _write_new_file(output, record.canonical_bytes())
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report({'status': 'PREPARED', 'output': str(output), 'packet_sha256': record.fingerprint(), 'scientific_qualification_performed': False})

    @campaign_app.command('rc-challenge-result')
    @retained_command('campaign rc-challenge-result', closed_output=True, native=False)
    def rc_challenge_result(
        run_id: Annotated[str, typer.Option('--run-id')],
        task_id: Annotated[str, typer.Option('--task-id')],
        receipt_id: Annotated[str, typer.Option('--receipt-id')],
        output_artifact_id: Annotated[str, typer.Option('--output-artifact-id')],
        result_id: Annotated[str, typer.Option('--result-id')],
        output: Annotated[Path, typer.Option('--output', help='New receipt-bound prerequisite/result file.')],
        issued_study_id: Annotated[str | None, typer.Option('--issued-study-id')] = None,
        execution_authority_id: Annotated[str | None, typer.Option('--execution-authority-id')] = None,
        reveal_authority_id: Annotated[str | None, typer.Option('--reveal-authority-id')] = None,
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Select an exact RC result; protected outcomes require actual current grants."""
        from empirical_lawhood.api.configuration import _write_new_file, preflight_new_configuration_output
        from empirical_lawhood.api.rc_challenge_results import bind_rc_challenge_selected_output, read_rc_challenge_result
        from empirical_lawhood.adapters.simulator_morphism_challenges.retained_results import RCChallengeResultAuthorityContext
        from empirical_lawhood.infrastructure.study_issue import ExternalStudyOperationAuthorityStore, ExternalIssuedStudyPublisher, PROGRAMME_ISSUE_GRANTEE_ID
        from empirical_lawhood.kernel.provenance import ObjectIdentity
        try:
            _, _, plane = _storage()
            output = preflight_new_configuration_output(output)
            identities = (issued_study_id, execution_authority_id, reveal_authority_id)
            if any(identities) and not all(identities):
                raise ValueError('Protected result selection requires all three current issue/grant IDs')
            context = None
            if all(identities):
                store = ExternalStudyOperationAuthorityStore(plane)
                issued = ExternalIssuedStudyPublisher(artifact_plane=plane, authority_store=store, grantee_id=PROGRAMME_ISSUE_GRANTEE_ID).load_executable_study(issued_study_id).manifest
                execution, reveal = store.load(execution_authority_id), store.load(reveal_authority_id)
                context = RCChallengeResultAuthorityContext(f'{result_id}.authority-context', ObjectIdentity.from_record(issued.issue_id, issued), ObjectIdentity.from_record(execution.authority_id, execution), ObjectIdentity.from_record(reveal.authority_id, reveal))
            attempt_stage('exact_current_result_selection', native_contact='none')
            record = bind_rc_challenge_selected_output(result_id=result_id, run_id=run_id, task_id=task_id, receipt_id=receipt_id, output_artifact_id=output_artifact_id, writer=plane, authority_context=context)
            view = read_rc_challenge_result(record.canonical_bytes(), writer=plane)
            _write_new_file(output, record.canonical_bytes())
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report({'status': 'RETAINED', 'output': str(output), 'result': view.summary})

    @campaign_app.command('finite-response-result')
    @retained_command('campaign finite-response-result', closed_output=True, native=False)
    def finite_response_result(
        run_id: Annotated[str, typer.Option('--run-id')],
        issued_study_id: Annotated[str, typer.Option('--issued-study-id')],
        execution_authority_id: Annotated[str, typer.Option('--execution-authority-id')],
        reveal_authority_id: Annotated[str, typer.Option('--reveal-authority-id')],
        task_id: Annotated[str, typer.Option('--task-id')],
        receipt_id: Annotated[str, typer.Option('--receipt-id')],
        result_kind: Annotated[str, typer.Option('--result-kind', help='calibration-native, qualification, closeout, root-reveal, root-inference, cohort or terminal')],
        config_id: Annotated[str, typer.Option('--config-id')],
        output: Annotated[Path, typer.Option('--output', help='New authenticated current result input.')],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Read an exact current scientific result under its persisted issue/reveal chain."""
        from empirical_lawhood.api.configuration import _write_new_file, preflight_new_configuration_output
        from empirical_lawhood.api.finite_rerun_authoring import current_result_input_from_receipt, read_finite_response_current_result, finite_response_current_result_summary
        try:
            _, _, plane = _storage()
            output = preflight_new_configuration_output(output)
            attempt_stage('exact_current_result_selection', native_contact='none')
            config = current_result_input_from_receipt(config_id=config_id, run_id=run_id, issued_study_id=issued_study_id, execution_authority_id=execution_authority_id, reveal_authority_id=reveal_authority_id, task_id=task_id, receipt_id=receipt_id, result_kind=result_kind, artifact_writer=plane)
            record = read_finite_response_current_result(config=config, artifact_writer=plane)
            summary = finite_response_current_result_summary(config=config, record=record)
            _write_new_file(output, config.canonical_bytes())
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report({'status': 'AUTHENTICATED', 'output': str(output), 'payload_schema': record.SCHEMA, 'payload_sha256': record.fingerprint(), 'result': summary, 'scientific_execution_performed': False})

    @campaign_app.command('finite-response-parent')
    @retained_command('campaign finite-response-parent', closed_output=True, native=False)
    def finite_response_parent(
        parent_id: Annotated[str, typer.Option('--parent-id')],
        native_result: Annotated[Path, typer.Option('--native-result', exists=True, dir_okay=False)],
        output: Annotated[Path, typer.Option('--output')],
        qualification_result: Annotated[Path | None, typer.Option('--qualification-result', exists=True, dir_okay=False)] = None,
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Join authenticated current calibration and qualification without importing old grants."""
        from empirical_lawhood.api.configuration import _write_new_file, preflight_new_configuration_output
        from empirical_lawhood.api.finite_rerun_authoring import current_parent_from_results
        try:
            _, _, plane = _storage()
            output = preflight_new_configuration_output(output)
            native = _configuration(native_result, 'finite-current-result')
            qualification = None if qualification_result is None else _configuration(qualification_result, 'finite-current-result')
            attempt_stage('exact_current_parent_join', native_contact='none')
            record = current_parent_from_results(parent_id=parent_id, native_result=native, qualification_result=qualification, artifact_writer=plane)
            _write_new_file(output, record.canonical_bytes())
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report({'status': 'AUTHENTICATED', 'output': str(output), 'parent_sha256': record.fingerprint(), 'authority_created': False})

    @campaign_app.command('original-f-export')
    @retained_command('campaign original-f-export', closed_output=True, native=False)
    def original_f_export(
        output_dir: Annotated[Path, typer.Option('--output-dir', help='New destination for the original F and its exact provenance files.')],
        payload_id: Annotated[str, typer.Option('--payload-id', help='Distinct current operand ID.')],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Deliver the installed original numerical law without fitting or qualification."""
        from empirical_lawhood.api.finite_operands import original_f_operand_export
        try:
            attempt_stage('original_operand_export', native_contact='none')
            record = original_f_operand_export(_destination(output_dir), payload_id=payload_id)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report({'status': 'EXPORTED', 'identity': record.identity.to_document(), 'output_dir': str(output_dir), 'scientific_execution_performed': False})

    @campaign_app.command('original-f-check')
    @retained_command('campaign original-f-check', closed_output=False, native=False)
    def original_f_check(
        operand: Annotated[Path, typer.Option('--operand', exists=True, dir_okay=False)],
        source_directory: Annotated[Path, typer.Option('--source-directory', exists=True, file_okay=False)],
        output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
    ) -> None:
        """Authenticate the selected original law and all exact provenance bytes."""
        from empirical_lawhood.api.finite_operands import original_f_operand_import
        try:
            attempt_stage('original_operand_authentication', native_contact='none')
            record = original_f_operand_import(_selected_input(operand), source_directory=source_directory)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report({'status': 'AUTHENTICATED', 'identity': record.identity.to_document(), 'qualification_performed': False})

    @campaign_app.command('nomination-export')
    @retained_command('campaign nomination-export', closed_output=True, native=False)
    def nomination_export(
        output_dir: Annotated[Path, typer.Option('--output-dir', help='New directory for the original and current coefficient transports.')],
        nomination_id: Annotated[str, typer.Option('--nomination-id')],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Deliver the frozen 48-root nomination for subsequent new calibration."""
        from empirical_lawhood.api.finite_operands import current_nomination_operand_export
        try:
            attempt_stage('nomination_export', native_contact='none')
            record = current_nomination_operand_export(_destination(output_dir), nomination_id=nomination_id)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report({'status': 'EXPORTED', 'identity': record.identity.to_document(), 'output_dir': str(output_dir), 'refitting_performed': False})

    @campaign_app.command('nomination-check')
    @retained_command('campaign nomination-check', closed_output=False, native=False)
    def nomination_check(
        operand: Annotated[Path, typer.Option('--operand', exists=True, dir_okay=False)],
        source_directory: Annotated[Path, typer.Option('--source-directory', exists=True, file_okay=False)],
        output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
    ) -> None:
        """Authenticate both physical namespaces and their exact numerical crosswalk."""
        from empirical_lawhood.api.finite_operands import current_nomination_operand_import
        try:
            attempt_stage('nomination_authentication', native_contact='none')
            record = current_nomination_operand_import(_selected_input(operand), source_directory=source_directory)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report({'status': 'AUTHENTICATED', 'identity': record.identity.to_document(), 'refitting_performed': False})

    @campaign_app.command('preparation-source-export')
    @retained_command('campaign preparation-source-export', closed_output=True, native=True)
    def preparation_source_export(
        config: Annotated[Path, typer.Option('--config', exists=True, dir_okay=False)],
        allocation: Annotated[Path, typer.Option('--allocation', exists=True, dir_okay=False)],
        original_f: Annotated[Path, typer.Option('--original-f', exists=True, dir_okay=False)],
        source_directory: Annotated[Path, typer.Option('--source-directory', exists=True, file_okay=False)],
        output_dir: Annotated[Path, typer.Option('--output-dir')],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Acquire the explicitly selected current 24-root development source."""
        from empirical_lawhood.api.finite_operands import original_f_operand_import, current_preparation_source_export
        try:
            root, _, plane = _storage()
            request = _configuration(config, 'preparation-source')
            roots = _configuration(allocation, 'matrix-allocation')
            lower = original_f_operand_import(_selected_input(original_f), source_directory=source_directory)
            from empirical_lawhood.api.matrix_numerical_provenance import capture_matrix_provenance
            capture_matrix_provenance(project_root=root, expected_code_sha256=request.code_sources_sha256,
                expected_lock_sha256=request.dependency_lock_sha256)
            destination = _destination(output_dir, plane)
            attempt_stage('native_acquisition', native_contact='started')
            record = current_preparation_source_export(destination, config=request, allocation=roots, original_f=lower, project_root=root, artifact_writer=plane)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report({'status': 'RETAINED', 'operand': record.to_document(), 'output_dir': str(output_dir), 'qualification': 'NONE'})

    @campaign_app.command('preparation-operands-export')
    @retained_command('campaign preparation-operands-export', closed_output=True, native=False)
    def preparation_operands_export(
        manifest: Annotated[Path, typer.Option('--manifest', exists=True, dir_okay=False)],
        arrays: Annotated[Path, typer.Option('--arrays', exists=True, dir_okay=False)],
        allocation: Annotated[Path, typer.Option('--allocation', exists=True, dir_okay=False)],
        original_f: Annotated[Path, typer.Option('--original-f', exists=True, dir_okay=False)],
        source_directory: Annotated[Path, typer.Option('--source-directory', exists=True, file_okay=False)],
        output_dir: Annotated[Path, typer.Option('--output-dir')],
        operand_id: Annotated[str, typer.Option('--operand-id')],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Fit and export reusable bridge/readiness operands from authenticated saved arrays."""
        from empirical_lawhood.api.finite_operands import original_f_operand_import, preparation_operands_export as export
        try:
            root, _, plane = _storage()
            from empirical_lawhood.api.matrix_numerical_provenance import capture_matrix_provenance
            capture_matrix_provenance(project_root=root)
            lower = original_f_operand_import(_selected_input(original_f), source_directory=source_directory)
            attempt_stage('saved_operand_fitting', native_contact='none')
            record = export(_destination(output_dir, plane), native_manifest_path=_selected_input(manifest, copy=False), native_arrays_path=_selected_input(arrays, maximum_bytes=64 * 1024**2, copy=False), allocation_path=_selected_input(allocation, copy=False), original_f=lower, operand_id=operand_id, artifact_writer=plane, project_root=root)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report({'status': 'EXPORTED', 'operand': record.to_document(), 'output_dir': str(output_dir), 'qualification': 'NONE'})

    @campaign_app.command('rc-challenge-source-export')
    @retained_command('campaign rc-challenge-source-export', closed_output=True, native=False)
    def rc_challenge_source_export(
        config: Annotated[Path, typer.Option('--config', exists=True, dir_okay=False)],
        output: Annotated[Path, typer.Option('--output', help='New canonical numeric-input file.')],
        prior_source: Annotated[list[Path] | None, typer.Option('--prior-source', help='Every prior source declared by the exposure census.')] = None,
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Publish actual current numerical RC inputs and committed source/export receipts."""
        from empirical_lawhood.api.configuration import _write_new_file, preflight_new_configuration_output
        from empirical_lawhood.api.rc_challenge_inputs import publish_rc_source_export
        from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
        try:
            output = preflight_new_configuration_output(output)
            _, _, plane = _storage()
            request = _configuration(config, 'rc-challenge-source')
            payloads = {}
            selected = prior_source or []
            if len(selected) != len(request.exposure_census.prior_source_artifacts):
                raise ValueError('Supply the complete prior source census in its declared artifact order')
            for artifact, path in zip(request.exposure_census.prior_source_artifacts, selected, strict=True):
                payloads[artifact.artifact_id] = read_bounded_bytes(_selected_input(path, maximum_bytes=artifact.size_bytes), maximum_bytes=artifact.size_bytes)
            attempt_stage('numeric_source_publication', native_contact='none')
            record = publish_rc_source_export(request=request, writer=plane, prior_source_payloads=payloads)
            _write_new_file(output, record.canonical_bytes())
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report({'status': 'PUBLISHED', 'input_sha256': record.fingerprint(), 'output': str(output), 'native_execution_performed': False})

    @campaign_app.command('finite-response-author')
    @retained_command('campaign finite-response-author', closed_output=True, native=False)
    def finite_response_author(
        config: Annotated[Path, typer.Option('--config', exists=True, dir_okay=False)],
        exposure: Annotated[Path, typer.Option('--exposure', exists=True, dir_okay=False)],
        nomination_directory: Annotated[Path, typer.Option('--nomination-directory', exists=True, file_okay=False)],
        output_dir: Annotated[Path, typer.Option('--output-dir')],
        parent: Annotated[Path | None, typer.Option('--parent', help='Exact current parent publications, receipts and reveal authority.')] = None,
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Compile one current calibration, qualification or prospective evaluation act."""
        from empirical_lawhood.api.finite_rerun_authoring import author_finite_response_rerun
        try:
            root, _, plane = _storage()
            packet = _configuration(config, 'finite-rerun')
            census = _configuration(exposure, 'finite-current-exposure')
            current_parent = None if parent is None else _configuration(parent, 'finite-current-parent')
            destination = _destination(output_dir, plane)
            attempt_stage('current_candidate_compilation', native_contact='none')
            report = author_finite_response_rerun(root=root, packet=packet, exposure=census, nomination_directory=nomination_directory, output_dir=destination, artifact_writer=plane, parent=current_parent)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report(report)

    @campaign_app.command('matrix-source-export')
    @retained_command('campaign matrix-source-export', closed_output=True, native=True)
    def matrix_source_export(
        config: Annotated[Path, typer.Option('--config', exists=True, dir_okay=False)],
        allocation: Annotated[Path, typer.Option('--allocation', exists=True, dir_okay=False)],
        output_dir: Annotated[Path, typer.Option('--output-dir')],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Acquire a bounded declared Q2/CIR1 development roster using the shared native source."""
        from empirical_lawhood.api.finite_operands import matrix_preparation_source_export
        try:
            root, _, plane = _storage()
            request = _configuration(config, 'matrix-source')
            roots = _configuration(allocation, 'matrix-allocation')
            from empirical_lawhood.api.matrix_numerical_provenance import capture_matrix_provenance
            capture_matrix_provenance(project_root=root, expected_code_sha256=request.code_sources_sha256,
                expected_lock_sha256=request.dependency_lock_sha256)
            destination = _destination(output_dir, plane)
            attempt_stage('native_acquisition', native_contact='started')
            record = matrix_preparation_source_export(destination, config=request, allocation=roots, project_root=root, artifact_writer=plane)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report({'status': 'RETAINED', 'operand': record.to_document(), 'qualification': 'NONE', 'output_dir': str(output_dir)})

    @campaign_app.command('rc-challenge-author')
    @retained_command('campaign rc-challenge-author', closed_output=True, native=False)
    def rc_challenge_author(
        config: Annotated[Path, typer.Option('--config', exists=True, dir_okay=False)],
        output_dir: Annotated[Path, typer.Option('--output-dir')],
        numeric_input: Annotated[Path | None, typer.Option('--numeric-input', help='Exact current published source/export operand; unnecessary for canary.')] = None,
        prerequisite: Annotated[list[Path] | None, typer.Option('--prerequisite', help='Complete explicitly selected receipt-bound upstream results.')] = None,
        evaluation_config: Annotated[Path | None, typer.Option('--evaluation-config', help='Fixed next evaluation config used by development design freeze.')] = None,
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Compile one independently issued current RC phase with exact prerequisite custody."""
        from empirical_lawhood.adapters.simulator_morphism_challenges.prerequisites import phase_external_records
        from empirical_lawhood.api.authoring_handoff import preflight_authoring_output_directory
        from empirical_lawhood.api.rc_challenge_authoring import author_rc_challenge_phase
        from empirical_lawhood.api.rc_challenge_inputs import read_rc_challenge_numeric_input, MAXIMUM_RC_NUMERIC_INPUT_BYTES
        from empirical_lawhood.api.rc_challenge_results import read_rc_challenge_result
        from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
        try:
            root, _, plane = _storage()
            destination = preflight_authoring_output_directory(directory=output_dir,
                repo_root=root, artifact_writer=plane)
            request = _configuration(config, 'rc-challenge-phase')
            packet = None if numeric_input is None else read_rc_challenge_numeric_input(read_bounded_bytes(_selected_input(numeric_input, copy=False), maximum_bytes=MAXIMUM_RC_NUMERIC_INPUT_BYTES), writer=plane)
            retained = tuple(read_rc_challenge_result(read_bounded_bytes(_selected_input(path, copy=False), maximum_bytes=16 * 1024**2), writer=plane).retained for path in prerequisite or [])
            evaluation = None if evaluation_config is None else _configuration(evaluation_config, 'rc-challenge-phase')
            records = phase_external_records(config=request, prerequisite_custody=retained, evaluation_config=evaluation)
            attempt_stage('current_candidate_compilation', native_contact='none')
            report = author_rc_challenge_phase(root=root, config=request, external_records=records, output_dir=destination, numeric_input=packet, artifact_writer=plane, prerequisite_custody=retained)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report(report)

    @campaign_app.command('preparation-author')
    @retained_command('campaign preparation-author', closed_output=True, native=False)
    def preparation_author(
        config: Annotated[Path, typer.Option('--config', exists=True, dir_okay=False)],
        output_dir: Annotated[Path, typer.Option('--output-dir')],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Compile an ordinary D/E or constructed Q/E selection with current published operands."""
        from empirical_lawhood.api.configuration import load_configuration_record
        from empirical_lawhood.api.preparation_applicability import preflight_preparation_authoring_output
        try:
            root, _, plane = _storage()
            output_dir = preflight_preparation_authoring_output(output_dir=output_dir, root=root, artifact_writer=plane)
            selected = load_configuration_record(_selected_input(config))
            if selected.SCHEMA == 'empirical-lawhood/composition/preparation-applicability/selection':
                from empirical_lawhood.api.preparation_applicability import author_preparation_applicability
                author = author_preparation_applicability
            elif selected.SCHEMA == 'empirical-lawhood/composition/constructed-preparation-applicability/selection':
                from empirical_lawhood.api.constructed_preparation_applicability import author_constructed_preparation_applicability
                author = author_constructed_preparation_applicability
            else:
                raise ValueError('Preparation authoring requires its typed ordinary or constructed selection')
            attempt_stage('current_candidate_compilation', native_contact='none')
            report = author(root=root, selection=selected, output_dir=output_dir, artifact_writer=plane)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report(report)

    @campaign_app.command('preparation-screen')
    @retained_command('campaign preparation-screen', closed_output=True, native=False)
    def preparation_screen(
        development_directory: Annotated[Path, typer.Option('--development-directory', exists=True, file_okay=False)],
        evaluation_directory: Annotated[Path, typer.Option('--evaluation-directory', exists=True, file_okay=False)],
        original_f: Annotated[Path, typer.Option('--original-f', exists=True, dir_okay=False)],
        source_directory: Annotated[Path, typer.Option('--source-directory', exists=True, file_okay=False)],
        output_dir: Annotated[Path, typer.Option('--output-dir')],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Read the complete ordinary96 panels for headroom and retrospective prefix forecasts."""
        from empirical_lawhood.api.finite_operands import original_f_operand_import
        from empirical_lawhood.api.preparation_screens import ordinary_preparation_screens
        try:
            lower = original_f_operand_import(_selected_input(original_f), source_directory=source_directory)
            attempt_stage('saved_ordinary_screen', native_contact='none')
            report = ordinary_preparation_screens(development_directory=development_directory, evaluation_directory=evaluation_directory, original_f=lower, output_dir=output_dir)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report(report)

    @campaign_app.command('preparation-readiness')
    @retained_command('campaign preparation-readiness', closed_output=True, native=False)
    def preparation_readiness(
        manifest: Annotated[Path, typer.Option('--manifest', exists=True, dir_okay=False)],
        arrays: Annotated[Path, typer.Option('--arrays', exists=True, dir_okay=False)],
        allocation: Annotated[Path, typer.Option('--allocation', exists=True, dir_okay=False)],
        original_f: Annotated[Path, typer.Option('--original-f', exists=True, dir_okay=False)],
        source_directory: Annotated[Path, typer.Option('--source-directory', exists=True, file_okay=False)],
        output_dir: Annotated[Path, typer.Option('--output-dir')],
        report_id: Annotated[str, typer.Option('--report-id')],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Analyze saved current24-root readiness; reject actual known failures and keep follow-ons unentered."""
        from empirical_lawhood.api.finite_operands import original_f_operand_import
        from empirical_lawhood.api.preparation_readiness import current_preparation_readiness
        try:
            root, _, plane = _storage()
            lower = original_f_operand_import(_selected_input(original_f), source_directory=source_directory)
            attempt_stage('saved_readiness_analysis', native_contact='none')
            report = current_preparation_readiness(manifest_path=_selected_input(manifest, copy=False), arrays_path=_selected_input(arrays, maximum_bytes=64 * 1024**2, copy=False), allocation_path=_selected_input(allocation, copy=False), original_f=lower, output_dir=output_dir, report_id=report_id, artifact_writer=plane, project_root=root)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report(report)

    @campaign_app.command('finite-response-bind-runtime')
    @retained_command('campaign finite-response-bind-runtime', closed_output=True, native=False)
    def finite_response_bind_runtime(
        authoring_dir: Annotated[Path, typer.Option('--authoring-dir', exists=True, file_okay=False)],
        context: Annotated[Path, typer.Option('--context', exists=True, dir_okay=False)],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Bind actual operator execution/reveal authorities after issue; create no grant."""
        from empirical_lawhood.api.finite_rerun_authoring import bind_finite_response_runtime
        from empirical_lawhood.api.authoring_handoff import resolve_authoring_directory
        try:
            root, profile, plane = _storage()
            selected, _ = resolve_authoring_directory(authoring_dir, repo_root=root, storage_profile=profile, escape_message='Runtime binding must stay under selected guarded storage')
            record = _configuration(context, 'finite-runtime-context')
            attempt_stage('runtime_authority_binding', native_contact='none')
            identity = bind_finite_response_runtime(directory=selected, context=record, artifact_writer=plane)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report({'status': 'BOUND', 'identity': identity.to_document(), 'authority_created': False})

    @campaign_app.command('integration-proof')
    @retained_command('campaign integration-proof', closed_output=False, native=False)
    def integration_proof(
        authoring_dir: Annotated[Path, typer.Option('--authoring-dir', exists=True, file_okay=False)],
        output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
    ) -> None:
        """Prove the exact selected integration graph/factories/outputs with no execution."""
        from empirical_lawhood.api.integration_handoffs import load_integration_handoff, prove_integration_handoff
        try:
            root, profile, plane = _storage()
            attempt_stage('provider_and_graph_proof', native_contact='none')
            handoff = load_integration_handoff(directory=authoring_dir, root=root, storage_profile=profile, artifact_writer=plane)
            report = prove_integration_handoff(handoff)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error(error)
        _report(report)
