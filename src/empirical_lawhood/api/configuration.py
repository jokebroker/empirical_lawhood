"""Bounded configuration validation and explicit preparation, without execution.

Existing per-consumer parsers and constructors remain authoritative. This API
does not inspect external prerequisites, acquire sources or instantiate providers.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal, DecimalException
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import stat

from empirical_lawhood.adapters._bounded_files import read_bounded_contained
from empirical_lawhood.adapters._strict_json import loads_external_json
from empirical_lawhood.api.codecs import (
    AuthoringCodecError,
    MAX_AUTHORING_BYTES,
    _parse_json,
    _parse_yaml,
    _validate_parsed_structure,
    decode_record,
)
from empirical_lawhood.api.configuration_registry import (
    ConfigurationConsumer,
    consumer_by_id,
    consumers_for_schema,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import CanonicalRecord


class ConfigurationError(ValueError):
    """A config refusal with an honest owner message and optional field location."""

    def __init__(self, code: str, message: str, *, field: str | None = None):
        super().__init__(message)
        self.code = code
        self.field = field

    def to_document(self) -> dict[str, object]:
        return {"status": "REFUSED", "code": self.code, "message": str(self), "field": self.field}


def _absolute(path: Path) -> Path:
    if ".." in path.parts:
        raise ConfigurationError("UNSAFE_PATH", "configuration paths cannot traverse parent directories")
    return path if path.is_absolute() else Path.cwd() / path


def _select(schema_id: str, consumer: str | None) -> ConfigurationConsumer:
    if consumer is not None:
        try:
            selected = consumer_by_id(consumer)
        except ValueError as error:
            raise ConfigurationError("UNKNOWN_CONSUMER", str(error)) from error
        if selected.schema_id != schema_id:
            raise ConfigurationError("CONSUMER_SCHEMA_MISMATCH", "configuration root differs from the selected consumer")
        return selected
    candidates = consumers_for_schema(schema_id)
    if not candidates:
        raise ConfigurationError("UNKNOWN_SCHEMA", f"unknown configuration schema: {schema_id}")
    if len(candidates) != 1:
        choices = ", ".join(item.consumer for item in candidates)
        raise ConfigurationError("CONSUMER_REQUIRED", f"select --consumer explicitly from: {choices}")
    return candidates[0]


def _decimal_size(value: Decimal) -> int:
    if not value.is_finite():
        raise ConfigurationError("INVALID_DECIMAL", "tagged Decimal must be finite")
    if value.is_zero():
        return 1
    sign, digits, exponent = value.as_tuple()
    assert isinstance(exponent, int)
    # Strip only trailing coefficient zeros, without arithmetic normalization.
    count = len(digits)
    while count > 1 and digits[count - 1] == 0:
        count -= 1
        exponent += 1
    fractional = max(-exponent, 0)
    return sign + max(count + exponent, 1) + fractional + bool(fractional)


def _bound_decimal_expansion(document: object, maximum_bytes: int) -> None:
    stack = [document]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if set(value) == {"decimal"} and isinstance(value["decimal"], str):
                try:
                    size = _decimal_size(Decimal(value["decimal"]))
                except DecimalException as error:
                    raise ConfigurationError("INVALID_DECIMAL", "invalid tagged Decimal") from error
                if size > maximum_bytes:
                    raise ConfigurationError("CONSUMER_BYTE_LIMIT", "Decimal expansion exceeds the consumer byte limit")
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)


def _canonical_size(value: object, limit: int) -> int:
    """Bound the complete output before allocating any fixed-point Decimal text."""
    if isinstance(value, CanonicalRecord):
        return _canonical_size({
            "schema": value.SCHEMA, "version": value.VERSION,
            "value": {field.name: getattr(value, field.name) for field in fields(value)},
        }, limit)
    if isinstance(value, Enum):
        return _canonical_size(value.value, limit)
    if isinstance(value, Decimal):
        result = len('{"decimal":""}') + _decimal_size(value)
    elif isinstance(value, dict):
        if set(value) == {"decimal"} and isinstance(value["decimal"], str):
            return _canonical_size(Decimal(value["decimal"]), limit)
        result = 2
        for index, (key, child) in enumerate(value.items()):
            result += bool(index) + _string_size(key, limit=limit) + 1
            result += _canonical_size(child, limit)
            if result > limit:
                break
    elif isinstance(value, (tuple, list, frozenset)):
        result = 2
        for index, child in enumerate(value):
            result += bool(index) + _canonical_size(child, limit)
            if result > limit:
                break
    elif isinstance(value, str):
        result = _string_size(value, limit=limit)
    else:
        result = len(json.dumps(value, ensure_ascii=True, allow_nan=False))
    if result > limit:
        raise ConfigurationError("CONSUMER_BYTE_LIMIT", "canonical configuration exceeds the consumer byte limit")
    return result


def _string_size(value: str, *, limit: int | None = None) -> int:
    # Count ensure_ascii JSON escapes without constructing an oversized string.
    count = 2
    for character in value:
        codepoint = ord(character)
        if character in {'"', "\\", "\b", "\f", "\n", "\r", "\t"}:
            count += 2
        elif codepoint < 0x20 or 0x7F <= codepoint <= 0xFFFF:
            count += 6
        elif codepoint > 0xFFFF:
            count += 12
        else:
            count += 1
        if limit is not None and count > limit:
            break
    return count


def _helper_inventory(document: object) -> None:
    """Validate the inert example's shape, without authenticating member locators."""
    from empirical_lawhood.kernel.serialization import validate_sha256, validate_stable_id

    if not isinstance(document, dict) or set(document) != {"schema", "version", "value"}:
        raise ValueError("exposed inventory helper envelope fields differ")
    if document["version"] != "1.0.0":
        raise ValueError("exposed inventory helper version differs")
    value = document["value"]
    if not isinstance(value, dict) or set(value) != {"inventory_id", "evidence_role", "history_complete", "members"}:
        raise ValueError("exposed inventory helper fields differ")
    validate_stable_id(value["inventory_id"])
    if value["evidence_role"] != "SYNTHETIC_EXPOSED_NONPROMOTABLE" or value["history_complete"] is not False:
        raise ValueError("exposed inventory helper cannot assert complete history or authority")
    if not isinstance(value["members"], list):
        raise ValueError("exposed inventory helper members must be an array")
    for member in value["members"]:
        if not isinstance(member, dict) or set(member) != {"locator", "sha256"}:
            raise ValueError("exposed inventory helper member fields differ")
        from empirical_lawhood.kernel.serialization import validate_relative_locator
        validate_relative_locator(member["locator"])
        validate_sha256(member["sha256"])


@dataclass(frozen=True, slots=True)
class _ValidatedConfiguration:
    report: dict[str, object]
    consumer: ConfigurationConsumer
    canonical: bytes | None


def _validate(path: Path, *, consumer: str | None) -> _ValidatedConfiguration:
    selected = None
    if consumer is not None:
        try:
            selected = consumer_by_id(consumer)
        except ValueError as error:
            raise ConfigurationError("UNKNOWN_CONSUMER", str(error)) from error
        if selected.schema_id is None:
            raise ConfigurationError("HELPER_ONLY", "human provenance template is not a runner config; copy and edit it manually")
    absolute = _absolute(path)
    limit = MAX_AUTHORING_BYTES if selected is None else selected.authoring_maximum_bytes
    try:
        raw = read_bounded_contained(Path(absolute.anchor), absolute, maximum_bytes=limit)
    except (OSError, ValueError) as error:
        raise ConfigurationError("INPUT_READ_REFUSED", str(error)) from error
    try:
        text = raw.decode("utf-8")
        suffix = path.suffix.casefold()
        if suffix == ".json":
            # Root discovery retains the owning import grammar. Canonical
            # authoring roots receive the stricter lexical parser below.
            document = loads_external_json(raw)
            _validate_parsed_structure(document)
        elif suffix in {".yaml", ".yml"}:
            document = _parse_yaml(text)
        else:
            raise ConfigurationError("UNSUPPORTED_FORMAT", "configuration input must use .json, .yaml or .yml")
    except (UnicodeError, AuthoringCodecError, ValueError, RecursionError) as error:
        if isinstance(error, ConfigurationError):
            raise
        raise ConfigurationError("INVALID_SYNTAX", str(error), field=getattr(error, "field", None)) from error
    if not isinstance(document, dict) or not isinstance(document.get("schema"), str):
        raise ConfigurationError("ROOT_SCHEMA_REQUIRED", "configuration needs a declared schema; provenance templates are helper-only")
    selected = _select(document["schema"], consumer)
    if len(raw) > selected.authoring_maximum_bytes:
        raise ConfigurationError("AUTHORING_BYTE_LIMIT", "editable document exceeds its declared authoring byte limit")
    if suffix != ".json" and (selected.requires_canonical or selected.disposition != "editable"):
        raise ConfigurationError("UNSUPPORTED_FORMAT", "this consumer requires JSON; YAML is not its declared authoring format")
    canonical = None
    try:
        if selected.disposition == "import":
            if len(raw) > selected.maximum_bytes:
                raise ConfigurationError("CONSUMER_BYTE_LIMIT", "retained import exceeds its consumer byte limit")
            if selected.consumer == "prepared-prior-census":
                from empirical_lawhood.adapters.composition.prepared_response.native_authoring import _prior_census
                _prior_census(raw)
            elif selected.consumer == "finite-prior-census":
                from empirical_lawhood.adapters.composition.finite_response_law.native_input import _census
                _census(raw)
            else:
                raise ConfigurationError("UNSUPPORTED_CONSUMER", "import consumer has no declared owning parser")
        elif selected.disposition == "helper":
            if selected.consumer != "exposed-inventory":
                raise ConfigurationError("HELPER_ONLY", "helper has no runner validation contract")
            if len(raw) > selected.maximum_bytes:
                raise ConfigurationError("CONSUMER_BYTE_LIMIT", "helper exceeds its byte limit")
            _helper_inventory(document)
        else:
            record_type = selected.record_type()
            if suffix == ".json":
                document = _parse_json(text)
            if document.get("version") != record_type.VERSION:
                raise ConfigurationError("WRONG_REVISION", "configuration version differs from its typed owner", field="document.version")
            _bound_decimal_expansion(document, selected.maximum_bytes)
            _canonical_size(document, selected.maximum_bytes - 1)
            if selected.disposition == "retained":
                record = decode_canonical_bytes(raw, record_type, maximum_bytes=selected.maximum_bytes)
            else:
                record = decode_record(document, record_type)
            _canonical_size(record, selected.maximum_bytes - 1)
            canonical = record.canonical_bytes()
            if len(canonical) > selected.maximum_bytes:
                raise ConfigurationError("CONSUMER_BYTE_LIMIT", "canonical configuration exceeds the consumer byte limit")
    except (TypeError, ValueError, DecimalException, RecursionError) as error:
        if isinstance(error, ConfigurationError):
            raise
        code = "RETAINED_BYTES_REFUSED" if selected.disposition == "retained" else "OWNER_VALIDATION_REFUSED"
        if selected.disposition == "editable" and "failed semantic validation" in str(error):
            code = "SEMANTIC_REFUSAL"
        raise ConfigurationError(code, str(error), field=getattr(error, "field", None)) from error
    ready = raw == canonical if canonical is not None else None
    # Authoring-capable consumers read the original document, not prepared bytes.
    consumer_ready = len(raw) <= selected.maximum_bytes and (ready if selected.requires_canonical else True)
    report: dict[str, object] = {
        "status": "VALIDATED", "consumer": selected.consumer,
        "schema_id": selected.schema_id, "record_version": document.get("version"),
        "disposition": selected.disposition, "command": selected.command,
        "owner": selected.owner, "consumer_loader": selected.loader,
        "input_sha256": sha256(raw).hexdigest(), "input_bytes": len(raw),
        "canonical_sha256": sha256(canonical).hexdigest() if canonical is not None else None,
        "canonical_bytes": len(canonical) if canonical is not None else None,
        "canonical_byte_ready": ready, "consumer_byte_ready": consumer_ready,
        "consumer_requires_canonical": selected.requires_canonical,
        "consumer_maximum_bytes": selected.maximum_bytes,
        "authoring_maximum_bytes": selected.authoring_maximum_bytes,
        "preparation_allowed": selected.preparation_allowed,
        "preparation_required": selected.preparation_allowed and not consumer_ready,
        "external_prerequisites_checked": False, "authority_granted": False,
        "validation_scope": {"editable": "typed values", "retained": "typed values and exact bytes",
                             "import": "owning census import only", "helper": "documentary shape only"}[selected.disposition],
        "evidence_ceiling": "input values only; external source, custody, census completeness and qualification uninspected",
    }
    return _ValidatedConfiguration(report, selected, canonical)


def validate_configuration(path: Path, *, consumer: str | None = None) -> dict[str, object]:
    """Validate one bounded declared input, without writes or external contact."""
    return _validate(path, consumer=consumer).report


def load_configuration_record(
    path: Path, *, consumer: str | None = None
) -> CanonicalRecord:
    """Load a declared operation input through its existing validation policy.

    Editable values needing canonical preparation remain a first-stop refusal.
    Helpers have no executable record; import/retained roles keep their policy.
    """
    validated = _validate(path, consumer=consumer)
    if validated.canonical is None:
        raise ConfigurationError("NO_RECORD_CONSUMER", "this input has no typed operation record")
    if validated.report["preparation_required"]:
        raise ConfigurationError("PREPARATION_REQUIRED", "prepare a new canonical file before using this consumer")
    if not validated.report["consumer_byte_ready"]:
        raise ConfigurationError("CONSUMER_BYTE_LIMIT", "input exceeds its operation consumer byte limit")
    record_type = validated.consumer.record_type()
    assert validated.consumer.maximum_bytes is not None
    return decode_canonical_bytes(
        validated.canonical, record_type, maximum_bytes=validated.consumer.maximum_bytes
    )


def _open_real_output_parent(absolute: Path) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise ConfigurationError("OUTPUT_REFUSED", "platform cannot enforce no-symlink output creation")
    parent = -1
    try:
        parent = os.open(absolute.anchor, os.O_RDONLY | directory | nofollow)
        for component in absolute.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | directory | nofollow, dir_fd=parent)
            os.close(parent)
            parent = child
        return parent
    except BaseException:
        if parent >= 0:
            os.close(parent)
        raise


def preflight_new_configuration_output(path: Path) -> Path:
    """Refuse invalid destinations before upstream publication or protected reads.

    This check creates nothing. The writer independently walks and checks the
    path again when creating the output, so preflight grants no write authority.
    """
    absolute = _absolute(path)
    if not absolute.name:
        raise ConfigurationError("OUTPUT_REFUSED", "output must select a new file, not the filesystem root")
    parent = -1
    try:
        parent = _open_real_output_parent(absolute)
        try:
            os.stat(absolute.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return absolute
        raise FileExistsError("output already exists")
    except OSError as error:
        raise ConfigurationError("OUTPUT_REFUSED", "output must be a new file under an existing real directory; overwrite and symlinks are refused") from error
    finally:
        if parent >= 0:
            os.close(parent)


def _write_new_file(path: Path, payload: bytes) -> None:
    absolute = _absolute(path)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    parent = -1
    descriptor = -1
    created = None
    try:
        parent = _open_real_output_parent(absolute)
        descriptor = os.open(absolute.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o600, dir_fd=parent)
        created = os.fstat(descriptor)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as error:
        if created is not None and parent >= 0:
            try:
                observed = os.stat(absolute.name, dir_fd=parent, follow_symlinks=False)
                if stat.S_ISREG(observed.st_mode) and (observed.st_dev, observed.st_ino) == (created.st_dev, created.st_ino):
                    os.unlink(absolute.name, dir_fd=parent)
            except OSError:
                pass
        if isinstance(error, OSError):
            raise ConfigurationError("OUTPUT_REFUSED", "output must be a new file under an existing real directory; overwrite and symlinks are refused") from error
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent >= 0:
            os.close(parent)


def prepare_configuration(path: Path, output_file: Path, *, consumer: str | None = None) -> dict[str, object]:
    """Prepare an editable root to a new file; preserve input and existing files."""
    validated = _validate(path, consumer=consumer)
    if not validated.consumer.preparation_allowed:
        raise ConfigurationError("PREPARATION_FORBIDDEN", "retained, import and helper inputs must remain under their owning byte/custody contracts")
    assert validated.canonical is not None
    _write_new_file(output_file, validated.canonical)
    return {
        **validated.report, "status": "PREPARED", "output_file": str(_absolute(output_file)),
        "output_sha256": sha256(validated.canonical).hexdigest(),
    }
