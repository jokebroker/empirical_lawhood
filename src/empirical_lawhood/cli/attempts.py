"""Opt-in development retention and the early Click error boundary.

SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextvars import ContextVar
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any, TypeVar, get_type_hints

import typer
from typer.core import TyperGroup

from empirical_lawhood.infrastructure.bounded_io import BoundedFileIOError
from empirical_lawhood.infrastructure.development_attempts import DevelopmentAttempt, MAX_INPUT_COPY_BYTES


ATTEMPT_OUTPUT_OPTION = typer.Option("--output-dir", help="Optional fresh directory for this development invocation, report and bounded failure diagnostics.")
ATTEMPT_DIRECTORY_OPTION = typer.Option("--attempt-dir", help="Optional fresh operational attempt directory, separate from the existing scientific output location.")
_current: ContextVar[DevelopmentAttempt | None] = ContextVar("development_attempt", default=None)
_participants: dict[str, dict[str, object]] = {}
_PATH_ROLES = {"config", "profile", "study", "archive", "source_root", "source_checkout", "native_python", "plan", "design_packet", "prior_exposure", "model_bank", "assignment", "parent_manifest", "custody", "reveal_record", "analysis_record", "authoring_input", "discovery_file", "upstream_binding"}
_F = TypeVar("_F", bound=Callable[..., Any])


def participation() -> dict[str, dict[str, object]]:
    """Describe explicitly registered export routes; no effects or discovery inference."""
    return {name: dict(spec) for name, spec in sorted(_participants.items())}


def _matches(attempt: DevelopmentAttempt | None, operation: str, destination: Path | None) -> bool:
    return attempt is not None and destination is not None and attempt.record["operation"] == operation and attempt.path == Path(os.path.abspath(destination))


def _disjoint(destination: Path, scientific: Path | None) -> None:
    if scientific is None:
        return
    attempt, product = destination.resolve(), scientific.resolve()
    if attempt.is_relative_to(product) or product.is_relative_to(attempt):
        raise ValueError("operational attempt and scientific output locations must be disjoint")


def _start(destination: Path, operation: str, *, native: bool, scientific: Path | None = None) -> DevelopmentAttempt:
    try:
        _disjoint(destination, scientific)
        return DevelopmentAttempt(destination, operation, native=native)
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(f"No durable development attempt could be created ({type(error).__name__}); destination must be a fresh directory with an existing real parent. No fallback location was used.", err=True)
        raise typer.BadParameter("unusable development attempt destination", param_hint="--attempt-dir" if scientific is not None else "--output-dir") from error


def _finish(attempt: DevelopmentAttempt, code: int, error: BaseException | None) -> bool:
    try:
        attempt.finish(code, error)
    except BaseException as secondary:
        typer.echo(f"Development attempt export failed ({type(secondary).__name__}); operation completed={attempt.record['operation_completed']}. Previously retained files: {attempt.path}. Work was not rerun.", err=True)
        if isinstance(secondary, KeyboardInterrupt):
            raise SystemExit(130) from secondary
        return False
    finally:
        attempt.close()
    typer.echo(f"Development attempt retained at {attempt.path}", err=True)
    return attempt.record["exit_code"] == code


def _exit_code(error: BaseException) -> int:
    if isinstance(error, (KeyboardInterrupt,)):
        return 130
    value = getattr(error, "exit_code", getattr(error, "code", None))
    if value is None and type(error).__name__ == "Abort":
        return 1
    return value if isinstance(value, int) else 1


def _exception(error: BaseException) -> BaseException:
    if isinstance(error, SystemExit):
        return error.__context__ or error.__cause__ or error
    return error


def retained_command(operation: str, *, closed_output: bool = False, native: bool = False) -> Callable[[_F], _F]:
    """Wrap one invocation, never call its computation a second time."""
    option = "--attempt-dir" if closed_output else "--output-dir"
    _participants[operation] = {"option": option, "native": native, "closed_output": closed_output}

    def decorate(function: _F) -> _F:
        signature = inspect.signature(function, eval_str=True)

        @wraps(function)
        def invoke(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            destination = bound.arguments.get("attempt_dir" if closed_output else "output_dir")
            attempt = _current.get()
            if not _matches(attempt, operation, destination):
                attempt = None
            if destination is None:
                if _current.get() is None:
                    return function(*args, **kwargs)
                cleared = _current.set(None)
                try:
                    return function(*args, **kwargs)
                finally:
                    _current.reset(cleared)
            owned = attempt is None
            token = None
            if owned:
                scientific = bound.arguments.get("output_dir", bound.arguments.get("output")) if closed_output else None
                attempt = _start(Path(destination), operation, native=native, scientific=scientific)
                token = _current.set(attempt)
            assert attempt is not None
            attempt.record["handler_entered"] = True
            # Locators only for held/multi-input operands. No archive, authority,
            # outcome payload, environment or arbitrary argv is copied here.
            attempt.record["declared_input_locators"] = [
                {"role": name, "locator": str(value.absolute())[:4096]}
                for name, value in bound.arguments.items() if name in _PATH_ROLES and isinstance(value, Path)
            ]
            try:
                result = function(*args, **kwargs)
            except BaseException as error:
                attempt.failure_exception = _exception(error)
                if owned:
                    _finish(attempt, _exit_code(error), _exception(error))
                raise
            else:
                if owned and not _finish(attempt, 0, None):
                    raise typer.Exit(code=1)
                return result
            finally:
                if token is not None:
                    _current.reset(token)

        invoke.__signature__ = signature  # type: ignore[attr-defined]
        invoke.__annotations__ = get_type_hints(function, include_extras=True)
        return invoke  # type: ignore[return-value]

    return decorate


def attempt_input(path: Path, *, maximum_bytes: int, copy: bool = True, role: str = "config", consumer: str | None = None) -> Path:
    """Keep one owned small snapshot and pass its exact bytes to the real owner."""
    attempt = _current.get()
    if attempt is None:
        return path
    try:
        attempt.record["native_contact"] = "none"
        attempt.event("input_validation")
        attempt.capture_input(role, path, maximum_bytes=maximum_bytes, copy=False)
        attempt.record["last_completed_stage"] = "input_identity_recorded"
        if not copy:
            return path
        expected = None
        if copy and path.suffix.casefold() in {".json", ".yaml", ".yml"}:
            from empirical_lawhood.api.configuration import validate_configuration
            from empirical_lawhood.api.configuration_registry import CONSUMERS

            consumers = [item.consumer for item in CONSUMERS if item.command == attempt.record["operation"]]
            selected = consumer if consumer is not None else (consumers[0] if len(consumers) == 1 else None)
            # Invalid arbitrary payloads are never persisted as input.json.
            # Value-valid pretty JSON still reaches the real exact-byte reader.
            validation = validate_configuration(path, consumer=selected)
            attempt.record["input_validation"] = validation
            # Larger closed scientific inputs keep their bounded observed
            # identity; operational retention never expands its snapshot ceiling.
            copy = validation["disposition"] == "editable" and validation["input_bytes"] <= MAX_INPUT_COPY_BYTES
            expected = str(validation["input_sha256"])
        elif copy:
            copy = False
        snapshot = attempt.capture_input(role, path, maximum_bytes=maximum_bytes, copy=copy, expected_sha256=expected)
        attempt.record["last_completed_stage"] = "input_snapshot_validated" if copy else "input_identity_recorded"
        return snapshot
    except BoundedFileIOError as error:
        raise ValueError("development input snapshot refused its bounded regular-file contract") from error


def attempt_stage(stage: str, *, native_contact: str = "unknown") -> None:
    attempt = _current.get()
    if attempt is not None:
        attempt.record["native_contact"] = native_contact
        attempt.event(stage)


def retain_development_report(report: object, *, allow_nan: bool = True) -> None:
    attempt = _current.get()
    if attempt is not None:
        attempt.operation_completed()
        attempt.event("result_serialization")
        attempt.report(json.dumps(report, sort_keys=True, indent=2, allow_nan=allow_nan))


def emit_development_report(report: object, *, allow_nan: bool = True) -> None:
    attempt = _current.get()
    if attempt is not None:
        attempt.operation_completed()
        attempt.event("result_serialization")
    rendered = json.dumps(report, sort_keys=True, indent=2, allow_nan=allow_nan)
    if attempt is not None:
        attempt.report(rendered)
    typer.echo(rendered)


def _option_values(command: Any, args: Sequence[str], *, leaf: bool = False) -> tuple[dict[str, str], list[str]]:
    """Inspect option structure without conversion or file preconditions."""
    parameters = {option: param for param in command.params for option in getattr(param, "opts", [])}
    values: dict[str, str] = {}
    rest: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            rest.extend(args[index + 1:])
            break
        spelling, separator, inline = token.partition("=")
        parameter = parameters.get(spelling)
        if parameter is not None and not parameter.is_flag:
            if separator:
                values[spelling] = inline
            elif index + 1 < len(args) and not args[index + 1].startswith("--"):
                values[spelling] = args[index + 1]
                index += 1
        elif not token.startswith("-"):
            if leaf:
                rest.append(token)
            else:
                rest.extend(args[index:])
                break
        index += 1
    return values, rest


def _selection(command: Any, args: Sequence[str]) -> tuple[str, Path, bool, Path | None] | None:
    if any(value in ("--help", "-h") for value in args):
        return None
    if len(args) > 4096:
        return None
    names: list[str] = []
    current, remaining = command, list(args)
    while getattr(current, "commands", None):
        _, remaining = _option_values(current, remaining)
        if not remaining:
            return None
        name = remaining.pop(0)
        current = current.commands.get(name)
        if current is None:
            return None
        names.append(name)
    operation = " ".join(names)
    # A campaign group can also be invoked directly by a caller.
    if operation not in _participants and "campaign " + operation in _participants:
        operation = "campaign " + operation
    spec = _participants.get(operation)
    if spec is None:
        return None
    values, _ = _option_values(current, remaining, leaf=True)
    selected = values.get(str(spec["option"]))
    if not selected or len(selected) > 4096:
        return None
    scientific = values.get("--output-dir", values.get("--output")) if spec["closed_output"] else None
    return operation, Path(selected), bool(spec["native"]), None if scientific is None else Path(scientific)


class AttemptGroup(TyperGroup):
    """Retain usable opt-in destinations before Click parses any preconditions."""

    def main(self, args: Sequence[str] | None = None, **kwargs: Any) -> Any:
        arguments = list(sys.argv[1:] if args is None else args)
        selected = None if any(name.endswith("_COMPLETE") for name in os.environ) else _selection(self, arguments)
        if selected is None:
            if _current.get() is None:
                return super().main(args=arguments, **kwargs)
            cleared = _current.set(None)
            try:
                return super().main(args=arguments, **kwargs)
            finally:
                _current.reset(cleared)
        operation, destination, native, scientific = selected
        if _matches(_current.get(), operation, destination):
            return super().main(args=arguments, **kwargs)
        try:
            attempt = _start(destination, operation, native=native, scientific=scientific)
        except typer.BadParameter as error:
            if not kwargs.get("standalone_mode", True):
                raise
            error.show()
            raise SystemExit(error.exit_code) from error
        token = _current.set(attempt)
        try:
            result = super().main(args=arguments, **kwargs)
        except BaseException as error:
            code = _exit_code(error)
            if not attempt.record["handler_entered"]:
                attempt.record["native_contact"] = "none"
            good = _finish(attempt, code, None if code == 0 else _exception(error))
            if code == 0 and not good:
                raise SystemExit(1) from error
            raise
        else:
            code = result if isinstance(result, int) else 0
            if not _finish(attempt, code, attempt.failure_exception):
                if not kwargs.get("standalone_mode", True):
                    return 1
                raise SystemExit(1)
            return result
        finally:
            _current.reset(token)


__all__ = ["ATTEMPT_DIRECTORY_OPTION", "ATTEMPT_OUTPUT_OPTION", "AttemptGroup", "attempt_input", "attempt_stage", "emit_development_report", "participation", "retain_development_report", "retained_command"]
