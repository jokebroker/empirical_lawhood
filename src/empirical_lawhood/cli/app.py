"""Thin composition root for the single current operator workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from empirical_lawhood.cli.configuration import config_app
from empirical_lawhood.cli.workflows import workflow_app
from empirical_lawhood.cli.attempts import (
    ATTEMPT_DIRECTORY_OPTION,
    ATTEMPT_OUTPUT_OPTION,
    attempt_input,
    AttemptGroup,
    attempt_stage,
    retain_development_report,
    retained_command,
)

from empirical_lawhood.cli.platform import (
    OutputFormat,
    admission_app,
    atlas_app,
    authority_app,
    campaign_app,
    capability_app,
    catalog_app,
    control_app,
    dataset_app,
    db_app,
    doctor_command,
    law_app,
    set_cli_inputs,
    source_app,
    system_app,
)
from empirical_lawhood.cli.integrations import register_integration_commands

register_integration_commands(campaign_app)

app = typer.Typer(
    cls=AttemptGroup,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Current denominator-local response-law authoring and execution workflow.",
)
app.add_typer(system_app, name="system")
app.add_typer(campaign_app, name="campaign")
app.add_typer(capability_app, name="capability")
app.add_typer(catalog_app, name="catalog")
app.add_typer(dataset_app, name="dataset")
app.add_typer(db_app, name="db")
app.add_typer(authority_app, name="authority")
app.add_typer(law_app, name="law")
app.add_typer(atlas_app, name="atlas")
app.add_typer(admission_app, name="admission")
app.add_typer(control_app, name="control")
app.add_typer(source_app, name="source")
app.add_typer(workflow_app, name="workflow")
app.add_typer(config_app, name="config")
example_app = typer.Typer(
    cls=AttemptGroup,
    no_args_is_help=True,
    help="Packaged, unqualified local demonstrations.",
)
app.add_typer(example_app, name="example")


@app.callback()
def global_inputs(
    context: typer.Context,
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="Target source checkout for project and source-bound operations.",
        ),
    ] = None,
    operator_profile: Annotated[
        Path | None,
        typer.Option(
            "--operator-profile",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help='Typed operator storage profile. This input grants no authority.',
        ),
    ] = None,
    dataset_projection_trust: Annotated[
        Path | None,
        typer.Option(
            "--dataset-projection-trust",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Explicit local dataset projection trust reference.",
        ),
    ] = None,
    approval_checker_trust: Annotated[
        Path | None,
        typer.Option(
            "--approval-checker-trust",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Explicit local approval checker trust registry.",
        ),
    ] = None,
    reactor_authoring_dir: Annotated[
        Path | None,
        typer.Option(
            '--reactor-authoring-dir',
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="Exact generated fresh reactor authoring directory for campaign operations.",
        ),
    ] = None,
    circuit_authoring_dir: Annotated[
        Path | None,
        typer.Option(
            '--circuit-authoring-dir',
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="Exact generated RC authoring directory for campaign operations.",
        ),
    ] = None,
    electron_gas_authoring_dir: Annotated[
        Path | None,
        typer.Option(
            '--electron-gas-authoring-dir',
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="Exact generated uniform electron gas analytic authoring directory for campaign operations.",
        ),
    ] = None,
    authoring_dir: Annotated[
        Path | None,
        typer.Option('--authoring-dir', exists=True, file_okay=False, resolve_path=True,
                     help="Exact current integration authoring directory for compile, preissue, run, results and recovery."),
    ] = None,
    synthetic_material_authoring_dir: Annotated[
        Path | None,
        typer.Option(
            '--synthetic-material-authoring-dir',
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="Generated lattice-pairing method authoring directory for campaign operations.",
        ),
    ] = None,
) -> None:
    """Select explicit project and storage inputs for commands that need them."""

    context.obj = {
        "project_root": project_root,
        "operator_profile": operator_profile,
        "dataset_projection_trust": dataset_projection_trust,
        "approval_checker_trust": approval_checker_trust,
        'reactor_authoring_dir': reactor_authoring_dir,
        'circuit_authoring_dir': circuit_authoring_dir,
        'electron_gas_authoring_dir': electron_gas_authoring_dir,
        'synthetic_material_authoring_dir': synthetic_material_authoring_dir,
        'authoring_dir': authoring_dir,
    }
    set_cli_inputs(
        project_root=project_root,
        operator_profile=operator_profile,
        dataset_projection_trust=dataset_projection_trust,
        approval_checker_trust=approval_checker_trust,
        reactor_authoring_dir=reactor_authoring_dir,
        circuit_authoring_dir=circuit_authoring_dir,
        electron_gas_authoring_dir=electron_gas_authoring_dir,
        synthetic_material_authoring_dir=synthetic_material_authoring_dir,
        authoring_dir=authoring_dir,
    )


@example_app.command("reactor-prefix")
@retained_command("example reactor-prefix", closed_output=True, native=True)
def reactor_prefix_example(
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", help="Fresh directory for the typed panel and report."
        ),
    ],
    attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
) -> None:
    """Run the pinned public two-decision native reactor prefix."""

    from empirical_lawhood.examples.reactor_prefix import run_reactor_prefix

    try:
        attempt_stage("reactor_prefix_example")
        report = run_reactor_prefix(output_dir)
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(f"reactor-prefix failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    attempt_stage("calculation_completed", native_contact="occurred")
    retain_development_report(report)
    typer.echo(
        f"{report['label']}: {report['complete_branches']} branches, "
        f"{report['observed_deliveries']} deliveries; {output_dir / 'report.json'}"
    )


@example_app.command("rc-information")
@retained_command("example rc-information", closed_output=False, native=False)
def rc_information_example(
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True,
                     help="RC analytical input; see experiments/rc-information/guide.md."),
    ],
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Evaluate the common-input RC separation and independent bound check."""
    from empirical_lawhood.adapters.methods.rc_information.contracts import RCInformationConfig
    from empirical_lawhood.api.configuration import ConfigurationError, load_configuration_record
    from empirical_lawhood.api.rc_information import calculate_rc_information

    try:
        selected = attempt_input(config, maximum_bytes=65536)
        record = load_configuration_record(selected, consumer="rc-information")
        if not isinstance(record, RCInformationConfig):
            raise TypeError("RC information operation requires its declared config")
        attempt_stage("analytical_calculation", native_contact="none")
        report = calculate_rc_information(record)
    except (ArithmeticError, ConfigurationError, OSError, TypeError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    retain_development_report(report.to_document())
    typer.echo(report.canonical_bytes().decode("utf-8"))


@app.command("doctor")
def platform_doctor(
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output format: text or deterministic JSON."),
    ] = OutputFormat.TEXT,
    route: Annotated[
        str | None,
        typer.Option("--route", help="Inspect the selected documented environment (including rc-challenges, finite-response-law and preparation-applicability) without native contact."),
    ] = None,
) -> None:
    """Read-only environment, Git, catalog and external-storage diagnosis."""

    doctor_command(output_format, route)


def main() -> None:
    """Dispatch the current CLI."""

    app()


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
