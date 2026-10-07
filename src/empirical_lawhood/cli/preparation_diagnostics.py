"""Low-friction current diagnostic selection, analysis and receipt reinspection.

SPDX-License-Identifier: MPL-2.0
"""

from pathlib import Path
from typing import Annotated

import typer

from empirical_lawhood.cli.attempts import ATTEMPT_DIRECTORY_OPTION, retained_command, attempt_stage


def register_preparation_diagnostic_commands(app):
    from empirical_lawhood.cli import integrations as h
    from empirical_lawhood.cli.preparation_integrations import _new_output, _write

    @app.command("preparation-diagnostic-phase-prepare")
    @retained_command("campaign preparation-diagnostic-phase-prepare", closed_output=True, native=False)
    def phase_prepare(
        selection: Annotated[Path, typer.Option("--selection", exists=True, dir_okay=False)],
        recovery_index: Annotated[str, typer.Option("--recovery-index", help="Exact relative recovery-index locator returned by campaign status.")],
        run_id: Annotated[str, typer.Option("--run-id")],
        issued_study_id: Annotated[str, typer.Option("--issued-study-id")],
        execution_authority_id: Annotated[str, typer.Option("--execution-authority-id")],
        reveal_authority_id: Annotated[str, typer.Option("--reveal-authority-id")],
        output: Annotated[Path, typer.Option("--output")],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ):
        """Extract all exact receipt selections from one actual complete terminal closure."""
        from empirical_lawhood.api.preparation_diagnostics import prepare_preparation_diagnostic_phase
        from empirical_lawhood.api.preparation_result_access import preparation_result_access
        try:
            _, _, plane = h._storage()
            destination = _new_output(output, plane)
            value = h._configuration(selection, "constructed-preparation-applicability-selection")
            access = preparation_result_access(run_id=run_id, issued_study_id=issued_study_id,
                execution_authority_id=execution_authority_id, reveal_authority_id=reveal_authority_id, writer=plane)
            attempt_stage("exact_complete_diagnostic_phase_selection", native_contact="none")
            record = prepare_preparation_diagnostic_phase(selection=value, result_access=access,
                recovery_index_relative_path=recovery_index, artifact_writer=plane)
            _write(destination, record)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "PREPARED", "output": str(destination), "phase": value.stage.phase,
            "independent_roots": len(record.roots), "native_calls": 0})

    @app.command("preparation-diagnostics-prepare")
    @retained_command("campaign preparation-diagnostics-prepare", closed_output=True, native=False)
    def input_prepare(
        analysis_id: Annotated[str, typer.Option("--analysis-id")],
        qualification: Annotated[Path, typer.Option("--qualification", exists=True, dir_okay=False)],
        evaluation: Annotated[Path, typer.Option("--evaluation", exists=True, dir_okay=False)],
        bootstrap_seed_q: Annotated[str, typer.Option("--bootstrap-seed-q")],
        bootstrap_seed_e: Annotated[str, typer.Option("--bootstrap-seed-e")],
        output: Annotated[Path, typer.Option("--output")],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ):
        """Join fixed Q8/E32 contexts and explicit diagnostic bootstrap streams."""
        from empirical_lawhood.adapters.methods.constructed_preparation_applicability.diagnostic_records import PreparationDiagnosticInput
        from empirical_lawhood.api.preparation_diagnostics import current_preparation_diagnostic_source_sha256
        try:
            _, _, plane = h._storage()
            destination = _new_output(output, plane)
            q = h._configuration(qualification, "preparation-diagnostic-phase")
            e = h._configuration(evaluation, "preparation-diagnostic-phase")
            attempt_stage("fixed_diagnostic_input_preparation", native_contact="none")
            record = PreparationDiagnosticInput(analysis_id, q, e, current_preparation_diagnostic_source_sha256(),
                int(bootstrap_seed_q, 0), int(bootstrap_seed_e, 0))
            _write(destination, record)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "PREPARED", "output": str(destination), "input_sha256": record.fingerprint(), "native_calls": 0})

    @app.command("preparation-diagnostics")
    @retained_command("campaign preparation-diagnostics", closed_output=True, native=False)
    def analyze(
        input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
        relative_root: Annotated[str, typer.Option("--relative-root", help="Fresh diagnostic publication root under selected storage.")],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ):
        """Authenticate and replay all40 retained roots; publish separate diagnostics."""
        from empirical_lawhood.api.preparation_diagnostics import publish_preparation_diagnostics
        try:
            _, _, plane = h._storage()
            _new_output(Path(plane.root.contract.canonical_path) / relative_root, plane, directory=True)
            config = h._configuration(input, "preparation-diagnostics")
            attempt_stage("authenticated_complete_posthoc_reduction", native_contact="none")
            result = publish_preparation_diagnostics(config, artifact_writer=plane, relative_root=relative_root)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "OUTCOME_VISIBLE_DIAGNOSTICS", "output": result.materialization.relative_path,
            "report_sha256": result.materialization.physical_sha256, "native_calls": 0, "qualification": "NOT_ESTABLISHED"})

    @app.command("preparation-diagnostics-result")
    @retained_command("campaign preparation-diagnostics-result", closed_output=True, native=False)
    def result(
        input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
        relative_root: Annotated[str, typer.Option("--relative-root")],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ):
        """Reauthenticate retained diagnostic custody without reacquisition or refitting."""
        from empirical_lawhood.api.preparation_diagnostics import read_preparation_diagnostics
        try:
            _, _, plane = h._storage()
            config = h._configuration(input, "preparation-diagnostics")
            attempt_stage("retained_diagnostic_result_read", native_contact="none")
            summary = read_preparation_diagnostics(config, artifact_writer=plane, relative_root=relative_root)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report(summary)
