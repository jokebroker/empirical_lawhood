"""Read-only task discovery, projected from workflow and CLI contract owners.

SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import StrEnum
from typing import Annotated

import typer

from empirical_lawhood.api.workflows import Workflow, list_workflows, show_workflow
from empirical_lawhood.cli.metadata import COMMAND_METADATA, invocation_contracts


class WorkflowFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


workflow_app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Static public task inventory. No input reads, providers or native contact.",
)


def workflow_details(workflow: Workflow) -> dict[str, object]:
    """Join command facts at the outer layer, keeping the API CLI-independent."""
    metadata = {entry.command: entry for entry in COMMAND_METADATA}
    contracts = invocation_contracts()
    result = asdict(workflow)
    result["prerequisites_inspected"] = False
    result["operations"] = [
        {
            **asdict(operation),
            "metadata": asdict(metadata[operation.command]),
            "invocation": asdict(contracts[operation.command]),
        }
        for operation in workflow.operations
    ]
    return result


def render_workflow(workflow: Workflow) -> str:
    facts = workflow_details(workflow)
    lines = [
        f"{workflow.workflow_id}: {workflow.title}",
        f"Boundary: {workflow.kind}",
        f"Sources: {', '.join(workflow.substrates)}",
        f"Methods: {', '.join(workflow.methods)}",
        f"Environment: {workflow.environment} (not inspected)",
        f"Inputs: {workflow.input_availability}",
        f"Ceiling: {workflow.scientific_ceiling}",
        f"First stop: {workflow.first_missing_prerequisite}",
        f"Guide: {workflow.guide}",
        workflow.resources,
        "Operations:",
    ]
    for entry in facts["operations"]:
        lines.append(f"  {entry['command']} [{entry['boundary']}]")
        lines.append(f"    Effects: {entry['metadata']['effects']}")
        invocation = entry["invocation"]
        lines.append(
            f"    Project: {invocation['project']}; storage: {invocation['storage']}"
        )
        lines.append(
            f"    Output: {invocation['output']}; refusal convention: {invocation['errors']}"
        )
        lines.append(f"    Native requirement: {entry['metadata']['native_software']}")
        lines.append(f"    Native boundary: {entry['metadata']['native_status']}")
    lines.append("Checkout inputs:")
    for item in workflow.inputs:
        lines.append(f"  {item.path}: {item.role}; {item.schema_id}")
    lines.append(f"Verification owners: {', '.join(workflow.verification_owners)}")
    return "\n".join(lines)


@workflow_app.command("list")
def workflow_list(
    output_format: Annotated[
        WorkflowFormat, typer.Option("--format", help="Output format: text or JSON.")
    ] = WorkflowFormat.TEXT,
) -> None:
    """List the closed workflow inventory without checking readiness."""
    workflows = list_workflows()
    if output_format == WorkflowFormat.JSON:
        typer.echo(
            json.dumps(
                {"workflows": [workflow_details(w) for w in workflows]}, indent=2
            )
        )
    else:
        typer.echo("Static workflow inventory; prerequisites have not been inspected.")
        for workflow in workflows:
            typer.echo(f"{workflow.workflow_id}: {workflow.title} [{workflow.kind}]")
        typer.echo(
            "Use workflow show ID for commands, input roles and checkout-only guides."
        )


@workflow_app.command("show")
def workflow_show(
    workflow_id: Annotated[
        str, typer.Argument(help="Exact public workflow ID from workflow list.")
    ],
    output_format: Annotated[
        WorkflowFormat, typer.Option("--format", help="Output format: text or JSON.")
    ] = WorkflowFormat.TEXT,
) -> None:
    """Show one static workflow; grant no execution or scientific authority."""
    try:
        workflow = show_workflow(workflow_id)
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    if output_format == WorkflowFormat.JSON:
        typer.echo(json.dumps(workflow_details(workflow), indent=2))
    else:
        typer.echo(render_workflow(workflow))
