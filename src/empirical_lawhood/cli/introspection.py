"""Deterministic structural introspection for the installed Click command tree."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import PurePath
from typing import Any


@dataclass(frozen=True, slots=True)
class CliParameterFact:
    """One installed option or argument, without invoking its callback."""

    name: str
    parameter_kind: str
    declarations: tuple[str, ...]
    required: bool
    default: str | None
    help: str
    type_name: str
    nargs: int
    multiple: bool
    is_flag: bool


@dataclass(frozen=True, slots=True)
class CliCommandFact:
    """Canonical structural facts for one installed command path."""

    path: tuple[str, ...]
    command_kind: str
    help: str
    parameters: tuple[CliParameterFact, ...]

    @property
    def command(self) -> str:
        return " ".join(self.path)


def _canonical_default(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, PurePath):
        return json.dumps(value.as_posix(), ensure_ascii=True)
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
        return f"callable:{module}.{name}".removeprefix("callable:.")
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def _parameter_fact(parameter: Any) -> CliParameterFact:
    declarations: tuple[str, ...]
    help_text = ""
    multiple = bool(getattr(parameter, "multiple", False))
    is_flag = False
    if parameter.param_type_name == "option":
        declarations = tuple((*parameter.opts, *parameter.secondary_opts))
        help_text = parameter.help or ""
        is_flag = parameter.is_flag
        parameter_kind = "option"
    elif parameter.param_type_name == "argument":
        declarations = (parameter.human_readable_name,)
        parameter_kind = "argument"
    else:  # pragma: no cover - Click currently exposes only options/arguments here.
        declarations = (parameter.human_readable_name,)
        parameter_kind = type(parameter).__name__
    return CliParameterFact(
        name=parameter.name or parameter.human_readable_name,
        parameter_kind=parameter_kind,
        declarations=declarations,
        required=parameter.required,
        default=_canonical_default(parameter.default),
        help=help_text,
        type_name=parameter.type.name,
        nargs=parameter.nargs,
        multiple=multiple,
        is_flag=is_flag,
    )


def _is_group(command: object) -> bool:
    return isinstance(getattr(command, "commands", None), dict)


def canonical_click_tree(root: Any) -> tuple[CliCommandFact, ...]:
    """Return every installed non-root path in deterministic lexical order."""

    facts: list[CliCommandFact] = []

    def visit(command: Any, path: tuple[str, ...]) -> None:
        facts.append(
            CliCommandFact(
                path=path,
                command_kind="group" if _is_group(command) else "command",
                help=command.help or "",
                parameters=tuple(_parameter_fact(parameter) for parameter in command.params),
            )
        )
        if _is_group(command):
            for name, child in sorted(command.commands.items()):
                visit(child, (*path, name))

    if not _is_group(root):
        raise TypeError("the installed CLI root must be a Click group")
    for name, command in sorted(root.commands.items()):
        visit(command, (name,))
    return tuple(facts)


def tree_by_path(facts: tuple[CliCommandFact, ...]) -> dict[tuple[str, ...], CliCommandFact]:
    """Index canonical facts and reject duplicate command paths."""

    indexed: dict[tuple[str, ...], CliCommandFact] = {}
    for fact in facts:
        if fact.path in indexed:
            raise ValueError(f"duplicate CLI command path: {fact.command}")
        indexed[fact.path] = fact
    return indexed


def fact_to_mapping(fact: CliCommandFact) -> dict[str, Any]:
    """Return a JSON-compatible mapping used by the generated reference."""

    return {
        "command": fact.command,
        "command_kind": fact.command_kind,
        "help": fact.help,
        "parameters": [
            {
                "declarations": list(parameter.declarations),
                "default": parameter.default,
                "help": parameter.help,
                "is_flag": parameter.is_flag,
                "multiple": parameter.multiple,
                "name": parameter.name,
                "nargs": parameter.nargs,
                "parameter_kind": parameter.parameter_kind,
                "required": parameter.required,
                "type_name": parameter.type_name,
            }
            for parameter in fact.parameters
        ],
    }
