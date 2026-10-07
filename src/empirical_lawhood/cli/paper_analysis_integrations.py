"""Current paper workflows through existing typed input/storage/attempt owners.

SPDX-License-Identifier: MPL-2.0
"""

from pathlib import Path
from typing import Annotated

import typer

from empirical_lawhood.cli.attempts import ATTEMPT_DIRECTORY_OPTION, retained_command, attempt_stage


def _commit(root):
    from empirical_lawhood.infrastructure.bounded_process import run_bounded_command
    value = run_bounded_command(("git", "-C", str(root), "rev-parse", "HEAD"),
        timeout_seconds=5, maximum_stdout_bytes=41, maximum_stderr_bytes=1024)
    if value.returncode or len(value.stdout) != 41:
        raise ValueError("Development source requires its actual selected checkout identity")
    return value.stdout.decode().strip()


def _project():
    from empirical_lawhood.cli import platform
    from empirical_lawhood.infrastructure.source_origin import require_executing_target_source
    root = platform.CLI_PROJECT_ROOT
    if root is None:
        raise ValueError("Select --project-root for the current dependency lock and source")
    require_executing_target_source(root)
    return root


def register_paper_analysis_commands(app):
    from empirical_lawhood.cli import integrations as h
    from empirical_lawhood.cli.preparation_integrations import _new_output, _write

    @app.command("matrix-geometry-prepare")
    @retained_command("campaign matrix-geometry-prepare", closed_output=True, native=False)
    def geometry_prepare(config_id: Annotated[str, typer.Option("--config-id")],
                         master_seed: Annotated[str, typer.Option("--master-seed")],
                         output: Annotated[Path, typer.Option("--output")],
                         workers: Annotated[int, typer.Option("--workers")] = 1,
                         prior_exposed_seed_sha256: Annotated[list[str] | None, typer.Option("--prior-exposed-seed-sha256")] = None,
                         attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Prepare the complete fresh numeric scan allocation against current source/lock."""
        from hashlib import sha256
        from empirical_lawhood.api.configuration import preflight_new_configuration_output
        from empirical_lawhood.api.matrix_geometry import prepare_matrix_geometry_scan
        try:
            destination = preflight_new_configuration_output(output)
            root = _project()
            attempt_stage("explicit_geometry_allocation_preparation", native_contact="none")
            record = prepare_matrix_geometry_scan(config_id=config_id, master_seed=int(master_seed,0),
                environment_lock_sha256=sha256((root / "uv.lock").read_bytes()).hexdigest(), workers=workers,
                prior_exposed_seed_sha256s=tuple(prior_exposed_seed_sha256 or ()))
            _write(destination, record)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "PREPARED", "output": str(destination), "input_sha256": record.fingerprint(),
            "native_contact": "none", "eligibility_granted": False})

    @app.command("matrix-geometry-bind-qualification")
    @retained_command("campaign matrix-geometry-bind-qualification", closed_output=True, native=False)
    def geometry_bind(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                      qualification_input: Annotated[Path, typer.Option("--qualification-input", exists=True, dir_okay=False)],
                      output: Annotated[Path, typer.Option("--output")],
                      attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Bind exact retained numerical prerequisite custody to the selected scan input."""
        from empirical_lawhood.api.matrix_geometry import bind_matrix_geometry_qualification
        try:
            root, _, plane = h._storage()
            destination = _new_output(output, plane)
            value = h._configuration(input, "matrix-geometry")
            qualification = h._configuration(qualification_input, "matrix-geometry")
            attempt_stage("exact_numerical_prerequisite_binding", native_contact="none")
            record = bind_matrix_geometry_qualification(config=value, qualification_config=qualification,
                artifact_writer=plane, implementation_commit=_commit(root))
            _write(destination, record)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "PREPARED", "output": str(destination), "input_sha256": record.fingerprint(),
            "native_contact": "none", "grants_authority": False})

    @app.command("matrix-geometry-prove")
    @retained_command("campaign matrix-geometry-prove", closed_output=True, native=False)
    def geometry_prove(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                       attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Prove the complete closed scan allocation/resource census without native contact."""
        from empirical_lawhood.api.matrix_geometry import prove_matrix_geometry_scan
        try:
            value = h._configuration(input, "matrix-geometry")
            attempt_stage("full_geometry_resource_proof", native_contact="none")
            report = prove_matrix_geometry_scan(value)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report(report)

    @app.command("matrix-geometry-run")
    @retained_command("campaign matrix-geometry-run", closed_output=True, native=True)
    def geometry_run(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                     attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Acquire the selected development scan, reusing exact completed cell receipts."""
        from empirical_lawhood.api.matrix_geometry import run_matrix_geometry_scan
        try:
            root, _, plane = h._storage()
            value = h._configuration(input, "matrix-geometry")
            attempt_stage("current_geometry_execution_preflight", native_contact="not_entered")
            result = run_matrix_geometry_scan(config=value, artifact_writer=plane, project_root=root,
                implementation_commit=_commit(root), progress=lambda done,total:attempt_stage(
                    f"geometry_cells_{done}_of_{total}", native_contact="development_native"))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report(result["summary"])

    @app.command("matrix-geometry-result")
    @retained_command("campaign matrix-geometry-result", closed_output=True, native=False)
    def geometry_result(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Read the exact complete scan result without repeating acquisition."""
        from empirical_lawhood.api.matrix_geometry import read_matrix_geometry_result
        try:
            root, _, plane = h._storage()
            value = h._configuration(input, "matrix-geometry")
            attempt_stage("retained_geometry_receipt_read", native_contact="none")
            result = read_matrix_geometry_result(config=value, artifact_writer=plane, implementation_commit=_commit(root))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report(result["summary"])

    @app.command("matrix-geometry-conformance")
    @retained_command("campaign matrix-geometry-conformance", closed_output=True, native=True)
    def geometry_conformance(
        input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
        coordinate: Annotated[str, typer.Option("--coordinate", help="Seven comma-separated cell coordinates in allocation order.")],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ):
        """Execute one declared real numerical cell; this does not complete the scan."""
        from empirical_lawhood.api.matrix_geometry import run_matrix_geometry_conformance
        try:
            root, _, plane = h._storage()
            value = h._configuration(input, "matrix-geometry")
            cell = tuple(int(part) for part in coordinate.split(","))
            attempt_stage("geometry_cell_conformance_preflight", native_contact="not_entered")
            result = run_matrix_geometry_conformance(config=value, coordinate=cell, artifact_writer=plane,
                project_root=root, implementation_commit=_commit(root))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report(result)

    @app.command("matrix-geometry-qualification")
    @retained_command("campaign matrix-geometry-qualification", closed_output=True, native=True)
    def geometry_qualification(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                               attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Retain actual numerical prerequisite checks; no issued authority is created."""
        from empirical_lawhood.api.matrix_geometry import run_matrix_numerical_qualification
        try:
            root, _, plane = h._storage()
            value = h._configuration(input, "matrix-geometry")
            attempt_stage("matrix_numerical_qualification_preflight", native_contact="not_entered")
            result = run_matrix_numerical_qualification(config=value, artifact_writer=plane,
                project_root=root, implementation_commit=_commit(root))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"disposition": result["disposition"], "report_sha256": result["report"].fingerprint(),
            "relative_path": result["manifest"].materialization.relative_path, "grants_authority": False})

    @app.command("selected-parent-reconstruct")
    @retained_command("campaign selected-parent-reconstruct", closed_output=True, native=True)
    def parent_reconstruct(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                           attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Reconstruct the fixed exposed nominated parent; mismatches stop conditioning."""
        from empirical_lawhood.api.selected_events import reconstruct_nominated_selected_parent
        try:
            root, _, plane = h._storage()
            value = h._configuration(input, "selected-parent")
            attempt_stage("selected_nominated_parent_replay_preflight", native_contact="not_entered")
            parent, _ = reconstruct_nominated_selected_parent(config=value, artifact_writer=plane,
                project_root=root, implementation_commit=_commit(root))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"parent_sha256": parent.fingerprint(), "conditional_source_instance": parent.conditional_source_instance,
            "evidence_ceiling": "NON_PROMOTABLE"})

    @app.command("selected-parent-alternate")
    @retained_command("campaign selected-parent-alternate", closed_output=True, native=False)
    def parent_alternate(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                         geometry_input: Annotated[Path, typer.Option("--geometry-input", exists=True, dir_okay=False)],
                         rollout_id: Annotated[str, typer.Option("--rollout-id")],
                         attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Select an authenticated current alternate parent as its own conditional instance."""
        from empirical_lawhood.api.selected_events import select_alternate_current_parent
        try:
            root, _, plane = h._storage()
            value = h._configuration(input, "selected-parent")
            scan = h._configuration(geometry_input, "matrix-geometry")
            attempt_stage("explicit_alternate_parent_selection", native_contact="none")
            parent, _ = select_alternate_current_parent(config=value, geometry_config=scan, rollout_id=rollout_id,
                artifact_writer=plane, implementation_commit=_commit(root))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"parent_sha256": parent.fingerprint(), "conditional_source_instance": parent.conditional_source_instance,
            "evidence_ceiling": "NON_PROMOTABLE"})

    @app.command("selected-events-prepare")
    @retained_command("campaign selected-events-prepare", closed_output=True, native=False)
    def selected_prepare(config_id: Annotated[str, typer.Option("--config-id")],
                         parent_input: Annotated[Path, typer.Option("--parent-input", exists=True, dir_okay=False)],
                         null_input: Annotated[Path, typer.Option("--null-input", exists=True, dir_okay=False)],
                         master_seed: Annotated[str, typer.Option("--master-seed")],
                         output: Annotated[Path, typer.Option("--output")],
                         attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Prepare the seven checkpoint roles and complete explicit conditional/bridge streams."""
        from empirical_lawhood.api.selected_events import prepare_selected_event
        try:
            root, _, plane = h._storage()
            destination = _new_output(output, plane)
            parent = h._configuration(parent_input, "selected-parent")
            null = h._configuration(null_input, "matrix-geometry")
            attempt_stage("current_selected_event_input_preparation", native_contact="none")
            value = prepare_selected_event(config_id=config_id, parent_request=parent, null_reference=null,
                master_seed=int(master_seed,0), artifact_writer=plane, implementation_commit=_commit(root))
            _write(destination, value)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "PREPARED", "output": str(destination), "input_sha256": value.fingerprint(), "native_contact": "none"})

    @app.command("selected-events-prove")
    @retained_command("campaign selected-events-prove", closed_output=True, native=False)
    def selected_prove(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                       attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Authenticate selected context and prove complete resource bounds before branches."""
        from empirical_lawhood.api.selected_events import prove_selected_event
        try:
            root, _, plane = h._storage()
            value = h._configuration(input, "selected-events")
            attempt_stage("full_selected_event_resource_proof", native_contact="none")
            result = prove_selected_event(config=value, artifact_writer=plane, implementation_commit=_commit(root))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report(result)

    @app.command("selected-events-run")
    @retained_command("campaign selected-events-run", closed_output=True, native=True)
    def selected_run(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                     attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Execute new conditional branches with exact retained checkpoint/receipt recovery."""
        from empirical_lawhood.api.selected_events import run_selected_event
        try:
            root, _, plane = h._storage()
            value = h._configuration(input, "selected-events")
            attempt_stage("selected_event_execution_preflight", native_contact="not_entered")
            result = run_selected_event(config=value, artifact_writer=plane, project_root=root,
                implementation_commit=_commit(root), progress=lambda *counts:attempt_stage(
                    "selected_event_branch_checkpoint", native_contact="development_native"))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report(result["summary"])

    @app.command("selected-events-conformance")
    @retained_command("campaign selected-events-conformance", closed_output=True, native=True)
    def selected_conformance(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                             checkpoint_step: Annotated[int, typer.Option("--checkpoint-step")],
                             branch_index: Annotated[int, typer.Option("--branch-index")],
                             attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Execute one exposed conditional future with exact replay and retained recovery."""
        from empirical_lawhood.api.selected_events import run_selected_event_conformance
        try:
            root, _, plane = h._storage()
            value = h._configuration(input, "selected-events")
            attempt_stage("selected_event_conformance_preflight", native_contact="not_entered")
            result = run_selected_event_conformance(config=value, checkpoint_step=checkpoint_step,
                branch_index=branch_index, artifact_writer=plane, project_root=root,
                implementation_commit=_commit(root))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        branch, replay = result["branch"], result["replay"]
        h._report({"branch_sha256": branch.fingerprint(), "checkpoint_step": branch.checkpoint_step,
            "branch_index": branch.branch_index, "terminal": branch.terminal.value,
            "seed_sha256": branch.seed_document_sha256, "replay_sha256": replay.fingerprint(),
            "parent_replay_exact": replay.parent_replay_exact, "checkpoint_restart_exact": replay.checkpoint_restart_exact,
            "evidence_ceiling": "NON_PROMOTABLE", "full_investigation_performed": False,
            "event_qualification_performed": False})

    @app.command("selected-events-result")
    @retained_command("campaign selected-events-result", closed_output=True, native=False)
    def selected_result(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Read all declared checkpoints, null denominators and conditional outcomes."""
        from empirical_lawhood.api.selected_events import read_selected_event_result
        try:
            root, _, plane = h._storage()
            value = h._configuration(input, "selected-events")
            attempt_stage("retained_selected_event_read", native_contact="none")
            result = read_selected_event_result(config=value, artifact_writer=plane, implementation_commit=_commit(root))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report(result["summary"])

    @app.command("matrix-history-allocation")
    @retained_command("campaign matrix-history-allocation", closed_output=True, native=False)
    def history_allocation(allocation_id: Annotated[str, typer.Option("--allocation-id")],
                           namespace: Annotated[str, typer.Option("--namespace")],
                           master_seed: Annotated[str, typer.Option("--master-seed")],
                           output: Annotated[Path, typer.Option("--output")],
                           proposed_unrun: Annotated[bool, typer.Option("--proposed-unrun")] = False,
                           attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Expand a numeric master into all256 history roots and four actual stream purposes."""
        from empirical_lawhood.api.configuration import preflight_new_configuration_output
        from empirical_lawhood.api.matrix_history_analysis import prepare_matrix_history_allocation
        try:
            destination = preflight_new_configuration_output(output)
            attempt_stage("explicit_history_allocation", native_contact="none")
            value = prepare_matrix_history_allocation(allocation_id=allocation_id, namespace=namespace,
                master_seed=int(master_seed,0), exposure="PROPOSED_UNRUN" if proposed_unrun else "EXPOSED_DEVELOPMENT_NONPROMOTABLE")
            _write(destination, value)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "ALLOCATED", "output": str(destination), "roots": len(value.roots), "eligibility_granted": False})

    @app.command("matrix-history-source-config")
    @retained_command("campaign matrix-history-source-config", closed_output=True, native=False)
    def history_source_config(config_id: Annotated[str, typer.Option("--config-id")],
                              allocation: Annotated[Path, typer.Option("--allocation", exists=True, dir_okay=False)],
                              output: Annotated[Path, typer.Option("--output")],
                              attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Bind edited allocations to current history source and dependency lock bytes."""
        from empirical_lawhood.api.configuration import preflight_new_configuration_output
        from empirical_lawhood.api.matrix_history_analysis import prepare_matrix_history_source_configuration
        try:
            destination = preflight_new_configuration_output(output)
            root = _project()
            value = h._configuration(allocation, "matrix-history-allocation")
            attempt_stage("current_history_source_config", native_contact="none")
            record = prepare_matrix_history_source_configuration(config_id=config_id, allocation=value, repo_root=root)
            _write(destination, record)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "PREPARED", "output": str(destination), "source_sha256": record.fingerprint()})

    @app.command("matrix-history-source-export")
    @retained_command("campaign matrix-history-source-export", closed_output=True, native=True)
    def history_source_export(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                              allocation: Annotated[Path, typer.Option("--allocation", exists=True, dir_okay=False)],
                              relative_root: Annotated[str, typer.Option("--relative-root")],
                              root_index: Annotated[list[int] | None, typer.Option("--root-index", help="Repeat for explicit incremental roots; default all 256.")] = None,
                              recover: Annotated[bool, typer.Option("--recover")] = False,
                              attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Acquire the complete exposed history roster, retaining each exact root separately."""
        from empirical_lawhood.api.matrix_history_analysis import produce_matrix_histories
        try:
            root, _, plane = h._storage()
            value = h._configuration(input, "matrix-history-source")
            roots = h._configuration(allocation, "matrix-history-allocation")
            attempt_stage("history_native_source_preflight", native_contact="not_entered")
            result = produce_matrix_histories(config=value, allocation=roots, writer=plane,
                relative_root=relative_root, repo_root=root, recover=recover,
                root_indices=tuple(range(256)) if root_index is None else tuple(root_index))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "EXPOSED_HISTORY_SOURCE", "relative_root": relative_root,
            "retained_roots": len(result), "qualification": "NOT_ESTABLISHED"})

    @app.command("matrix-history-import")
    @retained_command("campaign matrix-history-import", closed_output=True, native=False)
    def history_import(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                       allocation: Annotated[Path, typer.Option("--allocation", exists=True, dir_okay=False)],
                       operand: Annotated[Path, typer.Option("--operand", exists=True, dir_okay=False)],
                       arrays: Annotated[Path, typer.Option("--arrays", exists=True, dir_okay=False)],
                       relative_root: Annotated[str, typer.Option("--relative-root")],
                       attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Import explicit saved arrays as supplied exposed data, without native attestation."""
        from empirical_lawhood.api.matrix_history_analysis import import_matrix_history_arrays
        try:
            root, _, plane = h._storage()
            value = h._configuration(input, "matrix-history-source")
            roots = h._configuration(allocation, "matrix-history-allocation")
            attempt_stage("supplied_history_array_import", native_contact="none")
            result = import_matrix_history_arrays(operand_path=operand, arrays_path=arrays,
                config=value, allocation=roots, writer=plane, relative_root=relative_root, repo_root=root)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "SUPPLIED_EXPOSED_ARRAYS", "relative_root": relative_root,
            "receipt_sha256": result.fingerprint(), "qualification": "NOT_ESTABLISHED"})

    @app.command("matrix-history-analyze")
    @retained_command("campaign matrix-history-analyze", closed_output=True, native=False)
    def history_analyze(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                        allocation: Annotated[Path, typer.Option("--allocation", exists=True, dir_okay=False)],
                        source_root: Annotated[str, typer.Option("--source-root", help="Explicit complete source receipt root in selected storage.")],
                        relative_root: Annotated[str, typer.Option("--relative-root")],
                        recover: Annotated[bool, typer.Option("--recover")] = False,
                        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Analyze every assigned history; future evaluator operands cannot alter causal seals."""
        from empirical_lawhood.api.matrix_history_analysis import analyze_matrix_histories, matrix_history_result_summary
        from empirical_lawhood.kernel.serialization import validate_relative_locator
        try:
            root, _, plane = h._storage()
            validate_relative_locator(source_root)
            value = h._configuration(input, "matrix-history-analysis")
            roots = h._configuration(allocation, "matrix-history-allocation")
            paths = tuple(f"{source_root}/h{index:03d}/receipt.json" for index in range(len(roots.roots)))
            attempt_stage("authenticated_history_analysis", native_contact="none")
            result = analyze_matrix_histories(config=value, allocation=roots, source_receipt_paths=paths,
                writer=plane, relative_root=relative_root, repo_root=root, recover=recover)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report(matrix_history_result_summary(result))

    @app.command("matrix-history-result")
    @retained_command("campaign matrix-history-result", closed_output=True, native=False)
    def history_result(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                       relative_path: Annotated[str, typer.Option("--relative-path", help="Exact retained analysis receipt locator.")],
                       attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Reinspect committed history results and source custody without reacquisition."""
        from empirical_lawhood.api.matrix_history_analysis import read_matrix_history_analysis, matrix_history_result_summary
        try:
            _, _, plane = h._storage()
            value = h._configuration(input, "matrix-history-analysis")
            attempt_stage("retained_history_analysis_read", native_contact="none")
            result = read_matrix_history_analysis(writer=plane, relative_path=relative_path, config=value)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report(matrix_history_result_summary(result))

    @app.command("matrix-transient-analyze")
    @retained_command("campaign matrix-transient-analyze", closed_output=True, native=False)
    def transient(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                  native_dir: Annotated[Path, typer.Option("--native-dir", exists=True, file_okay=False)],
                  bridge_dir: Annotated[Path, typer.Option("--bridge-dir", exists=True, file_okay=False)],
                  original_f_dir: Annotated[Path, typer.Option("--original-f-dir", exists=True, file_okay=False)],
                  output: Annotated[Path, typer.Option("--output")],
                  attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Reuse current kernels/fold fits for complete transfer analysis and exploratory widths."""
        from empirical_lawhood.api.matrix_preparation_analysis import transient_response_analysis
        try:
            _, _, plane = h._storage()
            destination = _new_output(output, plane, directory=True)
            value = h._configuration(input, "matrix-transient-analysis")
            attempt_stage("current_saved_transient_analysis", native_contact="none")
            result = transient_response_analysis(destination, config=value, native_directory=native_dir,
                bridge_directory=bridge_dir, original_f_directory=original_f_dir, artifact_writer=plane, project_root=_project())
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": result.disposition, "output": str(destination), "report_sha256": result.fingerprint(), "qualification": "NOT_ESTABLISHED"})

    @app.command("matrix-baseline-analyze")
    @retained_command("campaign matrix-baseline-analyze", closed_output=True, native=False)
    def baseline(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                 transient_dir: Annotated[Path, typer.Option("--transient-dir", exists=True, file_okay=False)],
                 original_f_dir: Annotated[Path, typer.Option("--original-f-dir", exists=True, file_okay=False)],
                 output: Annotated[Path, typer.Option("--output")],
                 attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Evaluate signed decomposition and support from retained P08 operands without refitting."""
        from empirical_lawhood.api.matrix_preparation_analysis import baseline_support_analysis
        try:
            _, _, plane = h._storage()
            destination = _new_output(output, plane, directory=True)
            value = h._configuration(input, "matrix-baseline-analysis")
            attempt_stage("current_saved_baseline_analysis", native_contact="none")
            result = baseline_support_analysis(destination, config=value, transient_directory=transient_dir,
                original_f_directory=original_f_dir, artifact_writer=plane, project_root=_project())
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": result.disposition, "output": str(destination), "report_sha256": result.fingerprint(), "qualification": "NOT_ESTABLISHED"})

    @app.command("matrix-tangent-config")
    @retained_command("campaign matrix-tangent-config", closed_output=True, native=False)
    def tangent_config(config_id: Annotated[str, typer.Option("--config-id")],
                       namespace: Annotated[str, typer.Option("--namespace")],
                       master_seed: Annotated[str, typer.Option("--master-seed")],
                       output: Annotated[Path, typer.Option("--output")],
                       summary_context: Annotated[str, typer.Option("--summary-context")] = "all",
                       attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Prepare the full explicit tangent roster and ranks before outcomes."""
        from empirical_lawhood.api.configuration import preflight_new_configuration_output
        from empirical_lawhood.api.matrix_preparation_analysis import tangent_preparation_configuration
        try:
            destination = preflight_new_configuration_output(output)
            attempt_stage("explicit_tangent_allocation", native_contact="none")
            record = tangent_preparation_configuration(config_id=config_id, namespace=namespace,
                master_seed=int(master_seed,0), summary_context=summary_context, project_root=_project())
            _write(destination, record)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "PREPARED", "output": str(destination), "input_sha256": record.fingerprint(), "qualification": "NOT_ESTABLISHED"})

    @app.command("matrix-tangent-source-export")
    @retained_command("campaign matrix-tangent-source-export", closed_output=True, native=True)
    def tangent_source(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                       output: Annotated[Path, typer.Option("--output")],
                       attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Acquire current preparations using the declared tangent source/rank census."""
        from empirical_lawhood.api.matrix_preparation_analysis import tangent_preparation_source_export
        try:
            _, _, plane = h._storage()
            destination = _new_output(output, plane, directory=True)
            value = h._configuration(input, "matrix-tangent")
            attempt_stage("current_tangent_source_preflight", native_contact="not_entered")
            record = tangent_preparation_source_export(destination, config=value,
                artifact_writer=plane, project_root=_project(),
                progress=lambda *args:attempt_stage("tangent_context_completed", native_contact="development_native"))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "EXPOSED_TANGENT_SOURCE", "output": str(destination), "manifest_sha256": record.fingerprint(), "qualification": "NOT_ESTABLISHED"})

    @app.command("matrix-tangent-analyze")
    @retained_command("campaign matrix-tangent-analyze", closed_output=True, native=False)
    def tangent(input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
                source_dir: Annotated[Path, typer.Option("--source-dir", exists=True, file_okay=False)],
                output: Annotated[Path, typer.Option("--output")],
                attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Analyze the complete tangent census and noncompensating numerical/support stops."""
        from empirical_lawhood.api.matrix_preparation_analysis import tangent_response_analysis
        try:
            _, _, plane = h._storage()
            destination = _new_output(output, plane, directory=True)
            value = h._configuration(input, "matrix-tangent")
            attempt_stage("current_saved_tangent_analysis", native_contact="none")
            record = tangent_response_analysis(destination, config=value, source_directory=source_dir,
                                               artifact_writer=plane, project_root=_project())
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": record.disposition, "output": str(destination), "report_sha256": record.fingerprint(), "qualification": "NOT_ESTABLISHED"})

    @app.command("matrix-analysis-result")
    @retained_command("campaign matrix-analysis-result", closed_output=True, native=False)
    def analysis_result(directory: Annotated[Path, typer.Option("--directory", exists=True, file_okay=False)],
                        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None):
        """Authenticate retained tangent/transient/baseline outputs without recomputing science."""
        from empirical_lawhood.api.matrix_preparation_analysis import preparation_analysis_read
        try:
            _, _, plane = h._storage()
            attempt_stage("retained_matrix_analysis_read", native_contact="none")
            record, _, _ = preparation_analysis_read(directory, artifact_writer=plane)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": record.disposition, "report_sha256": record.fingerprint(),
            "metrics": {metric.name: None if metric.value is None else str(metric.value) for metric in record.metrics}, "qualification": "NOT_ESTABLISHED"})
