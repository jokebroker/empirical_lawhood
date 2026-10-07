"""Join installed CLI structure with exact documentation/effect annotations."""

from __future__ import annotations

from dataclasses import dataclass, replace
import shlex
from typing import Iterable

from empirical_lawhood.cli.introspection import (
    CliCommandFact,
    CliParameterFact,
    tree_by_path,
)
from empirical_lawhood.cli.metadata import CommandMetadata


class CliReferenceContractError(ValueError):
    """The installed tree, metadata or documented invocation is inconsistent."""


@dataclass(frozen=True, slots=True)
class CliReferenceEntry:
    fact: CliCommandFact
    metadata: CommandMetadata
    annotation_source: str


@dataclass(frozen=True, slots=True)
class ParsedCliExample:
    command: str
    path: tuple[str, ...]
    options: tuple[str, ...]
    contains_placeholders: bool
    tokens: tuple[str, ...]


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    items = tuple(values)
    return tuple(sorted({value for value in items if items.count(value) > 1}))


def join_command_metadata(
    facts: tuple[CliCommandFact, ...],
    group_metadata: tuple[CommandMetadata, ...],
    command_metadata: tuple[CommandMetadata, ...],
) -> tuple[CliReferenceEntry, ...]:
    """Build an exact-path join, inheriting only top-level group annotations."""

    indexed = tree_by_path(facts)
    group_duplicates = _duplicates(record.command for record in group_metadata)
    command_duplicates = _duplicates(record.command for record in command_metadata)
    if group_duplicates or command_duplicates:
        raise CliReferenceContractError(
            "duplicate CLI metadata paths: "
            f"groups={group_duplicates!r}, commands={command_duplicates!r}"
        )

    for record in (*group_metadata, *command_metadata):
        native, status = record.native_software, record.native_status
        if (native is None) != (status is None) or any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in (native, status)
        ):
            raise CliReferenceContractError(
                f"incomplete native prerequisite/status metadata: {record.command}"
            )

    top_level_paths = {fact.command for fact in facts if len(fact.path) == 1}
    declared_groups = {record.command for record in group_metadata}
    if top_level_paths != declared_groups:
        raise CliReferenceContractError(
            "top-level CLI metadata differs from the installed tree: "
            f"missing={sorted(top_level_paths - declared_groups)!r}, "
            f"stale={sorted(declared_groups - top_level_paths)!r}"
        )

    group_by_name = {record.command: record for record in group_metadata}
    command_by_path = {
        tuple(record.command.split()): record for record in command_metadata
    }
    installed_paths = set(indexed)
    stale_commands = sorted(
        " ".join(path) for path in set(command_by_path) - installed_paths
    )
    if stale_commands:
        raise CliReferenceContractError(
            f"installed CLI metadata names unavailable paths: {stale_commands!r}"
        )

    installed_groups = {
        record.command
        for record in group_metadata
        if record.lifecycle.startswith("installed")
    }
    installed_commands = {
        fact.path
        for fact in facts
        if fact.command_kind == "command" and fact.path[0] in installed_groups
    }
    declared_commands = set(command_by_path)
    if installed_commands != declared_commands:
        raise CliReferenceContractError(
            "installed leaf-command metadata differs from the installed tree: "
            f"missing={sorted(' '.join(path) for path in installed_commands - declared_commands)!r}, "
            f"stale={sorted(' '.join(path) for path in declared_commands - installed_commands)!r}"
        )

    entries: list[CliReferenceEntry] = []
    for fact in facts:
        exact = command_by_path.get(fact.path)
        if exact is not None:
            metadata = exact
            source = "exact-command"
        else:
            parent = group_by_name[fact.path[0]]
            metadata = replace(parent, command=fact.command)
            source = "exact-group" if len(fact.path) == 1 else "parent-group"
        entries.append(CliReferenceEntry(fact, metadata, source))
    return tuple(entries)


def _has_placeholder(value: str) -> bool:
    return ("<" in value and ">" in value) or ("{" in value and "}" in value)


def logical_shell_commands(body: str) -> tuple[str, ...]:
    """Collapse shell continuations and return logical, non-comment lines."""

    commands: list[str] = []
    pending = ""
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if line.startswith("$"):
            line = line[1:].lstrip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
            continue
        value = (pending + line).strip()
        pending = ""
        if value:
            commands.append(value)
    if pending:
        raise CliReferenceContractError("unterminated shell continuation")
    return tuple(commands)


def _cli_remainder(tokens: tuple[str, ...]) -> tuple[str, ...] | None:
    if tokens[:3] == ("uv", "run", "empirical-lawhood"):
        return tokens[3:]
    if tokens[:1] == ("empirical-lawhood",):
        return tokens[1:]
    return None


def _option_map(fact: CliCommandFact) -> dict[str, CliParameterFact]:
    return {
        declaration: parameter
        for parameter in fact.parameters
        if parameter.parameter_kind == "option"
        for declaration in parameter.declarations
    }


def parse_cli_example(
    command: str, facts: tuple[CliCommandFact, ...]
) -> ParsedCliExample | None:
    """Parse and structurally validate one documented ``empirical-lawhood`` invocation."""

    try:
        tokens = tuple(shlex.split(command, comments=True, posix=True))
    except ValueError as error:
        raise CliReferenceContractError(
            f"invalid shell command {command!r}: {error}"
        ) from error
    remainder = _cli_remainder(tokens)
    if remainder is None:
        return None
    indexed = tree_by_path(facts)
    if not remainder:
        return ParsedCliExample(command, (), (), False, tokens)
    if remainder[0] in {"--help", "-h"}:
        return ParsedCliExample(command, (), (remainder[0],), False, tokens)
    if remainder[0].startswith("-"):
        raise CliReferenceContractError(
            f"unknown root option in {command!r}: {remainder[0]}"
        )

    path: tuple[str, ...] = (remainder[0],)
    if path not in indexed:
        if _has_placeholder(remainder[0]):
            return ParsedCliExample(command, path, (), True, tokens)
        raise CliReferenceContractError(
            f"unknown CLI command path in {command!r}: {remainder[0]}"
        )
    position = 1
    fact = indexed[path]
    while fact.command_kind == "group" and position < len(remainder):
        token = remainder[position]
        if token.startswith("-"):
            break
        candidate = (*path, token)
        if candidate not in indexed:
            if _has_placeholder(token):
                return ParsedCliExample(command, candidate, (), True, tokens)
            raise CliReferenceContractError(
                f"unknown nested CLI command in {command!r}: {' '.join(candidate)}"
            )
        path = candidate
        fact = indexed[path]
        position += 1

    arguments = remainder[position:]
    option_map = _option_map(fact)
    seen_parameter_names: set[str] = set()
    seen_options: list[str] = []
    positional: list[str] = []
    contains_placeholders = any(_has_placeholder(token) for token in arguments)
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == "--":
            positional.extend(arguments[index + 1 :])
            break
        if token in {"--help", "-h"}:
            seen_options.append(token)
            index += 1
            continue
        if token.startswith("-"):
            declaration, separator, _inline_value = token.partition("=")
            parameter = option_map.get(declaration)
            if parameter is None:
                raise CliReferenceContractError(
                    f"unknown option for {' '.join(path)!r} in {command!r}: {declaration}"
                )
            seen_options.append(declaration)
            seen_parameter_names.add(parameter.name)
            index += 1
            if not parameter.is_flag and not separator:
                value_count = max(parameter.nargs, 1)
                if index + value_count > len(arguments):
                    raise CliReferenceContractError(
                        f"option {declaration!r} is missing a value in {command!r}"
                    )
                values = arguments[index : index + value_count]
                if any(
                    value in {"--help", "-h"} or value.startswith("-")
                    for value in values
                ):
                    raise CliReferenceContractError(
                        f"option {declaration!r} is missing a value in {command!r}"
                    )
                index += value_count
            continue
        positional.append(token)
        index += 1

    if "--help" not in seen_options and "-h" not in seen_options:
        missing_options = sorted(
            parameter.declarations[0]
            for parameter in fact.parameters
            if parameter.parameter_kind == "option"
            and parameter.required
            and parameter.name not in seen_parameter_names
        )
        required_arguments = sum(
            max(parameter.nargs, 1)
            for parameter in fact.parameters
            if parameter.parameter_kind == "argument" and parameter.required
        )
        if missing_options:
            raise CliReferenceContractError(
                f"required options absent for {' '.join(path)!r} in {command!r}: {missing_options!r}"
            )
        if len(positional) < required_arguments:
            raise CliReferenceContractError(
                f"required arguments absent for {' '.join(path)!r} in {command!r}"
            )
        if not any(parameter.multiple for parameter in fact.parameters) and positional:
            declared_arguments = sum(
                max(parameter.nargs, 1)
                for parameter in fact.parameters
                if parameter.parameter_kind == "argument"
            )
            if len(positional) > declared_arguments:
                raise CliReferenceContractError(
                    f"unexpected positional values for {' '.join(path)!r} in {command!r}: {positional!r}"
                )

    return ParsedCliExample(
        command=command,
        path=path,
        options=tuple(seen_options),
        contains_placeholders=contains_placeholders,
        tokens=tokens,
    )
