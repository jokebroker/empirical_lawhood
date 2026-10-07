"""Offline structural projections of closed configuration owners.

Runtime parsing/constructors remain the lexical and scientific authority.
Unsupported annotations fail generation instead of becoming unconstrained schemas.
"""

from __future__ import annotations

from dataclasses import fields
from decimal import Decimal
from enum import Enum
from hashlib import sha256
from importlib.resources import files
import json
import types
from typing import Union, get_args, get_origin

from empirical_lawhood.api.configuration_registry import CONSUMERS, ConfigurationConsumer
from empirical_lawhood.kernel.decoding import record_annotations
from empirical_lawhood.kernel.serialization import CanonicalRecord

RESOURCE_PACKAGE = "empirical_lawhood.resources.config_schemas"
DIALECT = "https://json-schema.org/draft/2020-12/schema"
_CENSUS_FIELDS = ("excluded_unit_ids", "excluded_seed_ids", "proposed_unit_ids", "proposed_seed_ids")
_UNITS = {"seconds": "s", "volts": "V", "amperes": "A", "farads": "F", "ohms": "ohm",
          "ms": "ms", "mv": "mV", "pa": "pA", "mohm": "Mohm"}


def _closed(properties: dict[str, object]) -> dict[str, object]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


class _Projection:
    def __init__(self) -> None:
        self.definitions: dict[str, object] = {}

    def record(self, record_type: type[CanonicalRecord]) -> dict[str, str]:
        key = record_type.__name__ + "_" + sha256(record_type.SCHEMA.encode()).hexdigest()[:12]
        if key not in self.definitions:
            self.definitions[key] = False  # Reserve before traversing recursive annotations.
            annotations = record_annotations(record_type)
            properties = {}
            for field in fields(record_type):
                projection = self.annotation(annotations[field.name])
                projection["description"] = field.name.replace("_", " ") + "; constructor checks semantics and units."
                suffix = field.name.rsplit("_", 1)[-1]
                if suffix in _UNITS:
                    projection["x-unit"] = _UNITS[suffix]
                properties[field.name] = projection
            self.definitions[key] = _closed({
                "schema": {"const": record_type.SCHEMA},
                "version": {"const": record_type.VERSION},
                "value": _closed(properties),
            })
        return {"$ref": f"#/$defs/{key}"}

    def annotation(self, annotation: object) -> dict[str, object]:
        origin, arguments = get_origin(annotation), get_args(annotation)
        if annotation is type(None):
            return {"type": "null"}
        if origin in {Union, types.UnionType}:
            return {"anyOf": [self.annotation(option) for option in arguments]}
        if origin is tuple:
            if len(arguments) == 2 and arguments[1] is Ellipsis:
                return {"type": "array", "items": self.annotation(arguments[0])}
            result: dict[str, object] = {"type": "array", "items": False,
                                        "minItems": len(arguments), "maxItems": len(arguments)}
            if arguments:
                result["prefixItems"] = [self.annotation(item) for item in arguments]
            return result
        if origin is frozenset and len(arguments) == 1:
            return {"type": "array", "items": self.annotation(arguments[0])}
        if annotation is Decimal:
            return _closed({"decimal": {
                "type": "string", "description": "Exact finite Decimal text; no binary float. Runtime bounds expansion and decides lexical/semantic acceptance.",
            }})
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            values = [item.value for item in annotation]
            if not all(isinstance(value, str) for value in values):
                raise TypeError(f"unsupported non-string enum: {annotation!r}")
            return {"type": "string", "enum": values}
        if isinstance(annotation, type) and issubclass(annotation, CanonicalRecord):
            return self.record(annotation)
        primitive = {str: "string", int: "integer", bool: "boolean"}
        if annotation in primitive:
            return {"type": primitive[annotation]}
        raise TypeError(f"unsupported configuration schema annotation: {annotation!r}")


def project_record(record_type: type[CanonicalRecord]) -> dict[str, object]:
    projection = _Projection()
    reference = projection.record(record_type)
    return {"$schema": DIALECT, **reference, "$defs": projection.definitions}


def _import_schema(consumer: ConfigurationConsumer) -> dict[str, object]:
    # Imports intentionally retain extensible historical metadata, exactly as
    # their existing readers do. They are never a preparation boundary.
    census = {field: {"type": "array", "items": {"type": "string"}} for field in _CENSUS_FIELDS}
    if consumer.consumer == "prepared-prior-census":
        for field in census.values():
            field["minItems"] = 1
        return {"$schema": DIALECT, "type": "object", "required": ["schema", "value"],
                "properties": {"schema": {"const": consumer.schema_id},
                               "value": {"type": "object", "properties": census, "required": list(census)}}}
    return {"$schema": DIALECT, "type": "object", "required": ["schema", *_CENSUS_FIELDS],
            "properties": {"schema": {"const": consumer.schema_id}, **census}}


def _helper_schema(consumer: ConfigurationConsumer) -> dict[str, object]:
    return {"$schema": DIALECT, **_closed({
        "schema": {"const": consumer.schema_id}, "version": {"const": "1.0.0"},
        "value": _closed({"inventory_id": {"type": "string"},
                          "evidence_role": {"const": "SYNTHETIC_EXPOSED_NONPROMOTABLE"},
                          "history_complete": {"const": False},
                          "members": {"type": "array", "items": _closed({
                              "locator": {"type": "string"},
                              "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                          })}}),
    })}


def generated_resources() -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    entries = []
    for consumer in CONSUMERS:
        filename = None
        if consumer.schema_id is not None:
            filename = consumer.consumer + ".schema.json"
            if consumer.disposition == "import":
                schema = _import_schema(consumer)
            elif consumer.disposition == "helper":
                schema = _helper_schema(consumer)
            else:
                schema = project_record(consumer.record_type())
            schema.update({
                "title": consumer.consumer.replace("-", " ").title(),
                "$id": "urn:empirical-lawhood:configuration:" + consumer.consumer,
                "description": "Structural projection only. Typed owner decides lexical/semantic acceptance; no external prerequisites or authority are checked.",
                "x-consumer": consumer.consumer, "x-disposition": consumer.disposition,
                "x-consumer-maximum-bytes": consumer.maximum_bytes,
                "x-preparation-allowed": consumer.preparation_allowed,
            })
            result[filename] = (json.dumps(schema, sort_keys=True, indent=2) + "\n").encode()
        entries.append({"consumer": consumer.consumer, "schema_id": consumer.schema_id,
                        "disposition": consumer.disposition, "schema_resource": filename,
                        "command": consumer.command, "owner": consumer.owner, "consumer_loader": consumer.loader,
                        "supplied_inputs": list(consumer.supplied_inputs),
                        "consumer_maximum_bytes": consumer.maximum_bytes,
                        "authoring_maximum_bytes": consumer.authoring_maximum_bytes,
                        "consumer_requires_canonical": consumer.requires_canonical,
                        "preparation_allowed": consumer.preparation_allowed})
    result["index.json"] = (json.dumps({"format_version": "1.0.0", "consumers": entries}, sort_keys=True, indent=2) + "\n").encode()
    return result


def configuration_schema(schema_id: str) -> dict[str, object]:
    """Read an allowlisted installed schema; never follow remote references."""
    root = files(RESOURCE_PACKAGE)
    index = json.loads(root.joinpath("index.json").read_text(encoding="utf-8"))
    candidates = [item for item in index["consumers"] if item["schema_id"] == schema_id]
    if len(candidates) != 1:
        raise ValueError(f"unknown or ambiguous configuration schema: {schema_id}")
    return json.loads(root.joinpath(candidates[0]["schema_resource"]).read_text(encoding="utf-8"))
