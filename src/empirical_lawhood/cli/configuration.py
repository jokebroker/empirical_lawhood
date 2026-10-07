"""No-contact configuration commands; root app registration stays separate."""

from __future__ import annotations

from enum import StrEnum
import json
from pathlib import Path
from typing import Annotated

import typer

from empirical_lawhood.api.configuration import ConfigurationError, prepare_configuration, validate_configuration
from empirical_lawhood.api.configuration_schemas import configuration_schema

config_app = typer.Typer(no_args_is_help=True, help="Validate editable inputs, prepare new canonical files, and inspect offline schemas.")


class ConfigurationFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


def _report(document: dict[str, object], output_format: ConfigurationFormat) -> None:
    if output_format is ConfigurationFormat.JSON:
        typer.echo(json.dumps(document, sort_keys=True, indent=2), err=document["status"] == "REFUSED")
    elif document["status"] == "REFUSED":
        typer.echo(f"{document['code']}: {document['message']}", err=True)
    else:
        typer.echo(f"{document['status']}: {document['consumer']} ({document['disposition']})")
        typer.echo(f"Input SHA256: {document['input_sha256']}")
        typer.echo(f"Consumer byte readiness: {document['consumer_byte_ready']}; canonical byte readiness: {document['canonical_byte_ready']}")
        if document.get("output_file"):
            typer.echo(f"Prepared file: {document['output_file']}\nOutput SHA256: {document['output_sha256']}")
        elif document["preparation_required"]:
            typer.echo("Values are valid; prepare a new file for this consumer.")
        typer.echo(str(document["evidence_ceiling"]))


@config_app.command("validate")
def config_validate(
    config: Annotated[Path, typer.Option("--config", help="Bounded JSON input; YAML only for declared authoring consumers.")],
    consumer: Annotated[str | None, typer.Option("--consumer", help="Explicit closed input role; required if the root is ambiguous.")] = None,
    output_format: Annotated[ConfigurationFormat, typer.Option("--format")] = ConfigurationFormat.TEXT,
) -> None:
    """Validate values and consumer byte readiness without execution or writes."""
    try:
        report = validate_configuration(config, consumer=consumer)
    except ConfigurationError as error:
        _report(error.to_document(), output_format)
        raise typer.Exit(code=2) from error
    _report(report, output_format)


@config_app.command("prepare")
def config_prepare(
    config: Annotated[Path, typer.Option("--config", help="Editable declared input.")],
    output_file: Annotated[Path, typer.Option("--output-file", help="New file under an existing real directory; no overwrite.")],
    consumer: Annotated[str | None, typer.Option("--consumer")] = None,
    output_format: Annotated[ConfigurationFormat, typer.Option("--format")] = ConfigurationFormat.TEXT,
) -> None:
    """Write canonical bytes for an editable root, preserving the original."""
    try:
        report = prepare_configuration(config, output_file, consumer=consumer)
    except ConfigurationError as error:
        _report(error.to_document(), output_format)
        raise typer.Exit(code=2) from error
    _report(report, output_format)


@config_app.command("schema")
def config_schema(schema_id: Annotated[str, typer.Option("--schema-id", help="Exact declared configuration root ID.")]) -> None:
    """Emit one packaged Draft2020-12 structural schema with local references."""
    try:
        schema = configuration_schema(schema_id)
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(schema, sort_keys=True, indent=2))
