"""Independent offline Draft2020-12 validation of the emitted projections."""

from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
import json
from pathlib import Path
from typing import ClassVar

from jsonschema import Draft202012Validator
import pytest
from referencing import Registry
from typer.testing import CliRunner

from empirical_lawhood.api.configuration import ConfigurationError, validate_configuration
from empirical_lawhood.api.configuration_registry import CONSUMERS
from empirical_lawhood.api.configuration_schemas import RESOURCE_PACKAGE, configuration_schema, generated_resources, project_record
from empirical_lawhood.cli.configuration import config_app
from empirical_lawhood.kernel.serialization import CanonicalRecord

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "experiments/rc-ladder-response/model.json"


def _offline(schema):
    def refuse(uri):
        raise AssertionError(f"schema attempted external resolution: {uri}")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, registry=Registry(retrieve=refuse))


def test_generated_resource_inventory_and_bytes_are_current():
    expected = generated_resources()
    resource = files(RESOURCE_PACKAGE)
    actual = {item.name for item in resource.iterdir() if item.name.endswith(".json")}
    assert actual == set(expected)
    for name, payload in expected.items():
        assert resource.joinpath(name).read_bytes() == payload


@pytest.mark.parametrize("consumer", [item for item in CONSUMERS if item.schema_id], ids=lambda item: item.consumer)
def test_emitted_schemas_accept_all_supplied_inputs_offline(consumer):
    validator = _offline(configuration_schema(consumer.schema_id))
    for locator in consumer.supplied_inputs:
        validator.validate(json.loads((ROOT / locator).read_bytes()))


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(extra=True),
    lambda d: d.update(version="9.0.0"),
    lambda d: d.pop("value"),
    lambda d: d["value"].update(extra=True),
    lambda d: d["value"].pop("scale_cells"),
    lambda d: d["value"].update(scale_cells=True),
    lambda d: d["value"].update(initial_voltages_volts=[False]),
    lambda d: d["value"]["action_segments"][0].update(schema="unknown/nested"),
    lambda d: d["value"]["action_segments"][0].update(version="9.0.0"),
    lambda d: d["value"]["action_segments"][0]["value"].update(extra=True),
    lambda d: d["value"]["left_source_resistance_ohms"].update(decimal=100),
    lambda d: d["value"]["left_source_resistance_ohms"].update(extra=True),
])
def test_structural_invalid_corpus_is_rejected_by_independent_validator(mutation):
    document = json.loads(RC.read_bytes())
    mutation(document)
    validator = _offline(configuration_schema(json.loads(RC.read_bytes())["schema"]))
    assert list(validator.iter_errors(document))


def test_structurally_valid_semantic_and_lexical_failures_remain_runtime_owned(tmp_path):
    document = json.loads(RC.read_bytes())
    validator = _offline(configuration_schema(document["schema"]))
    for value in (4.0, 0):
        document["value"]["scale_cells"] = value
        validator.validate(document)
        path = tmp_path / "input.json"
        path.write_text(json.dumps(document))
        with pytest.raises(ConfigurationError):
            validate_configuration(path)


class _Choice(StrEnum):
    FIRST = "first"
    SECOND = "second"


@dataclass(frozen=True)
class _TupleRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/test/configuration-tuple"
    pair: tuple[int, bool]
    empty: tuple[()]
    choice: _Choice
    nullable: str | None
    defaulted: int = 3


def test_fixed_tuples_enums_nullability_and_defaults_are_projected_exactly():
    validator = _offline(project_record(_TupleRecord))
    document = _TupleRecord((1, False), (), _Choice.FIRST, None).to_document()
    validator.validate(document)
    for field, wrong in (("pair", [1]), ("pair", [1, False, 2]), ("pair", [False, 1]),
                         ("empty", [0]), ("choice", "unknown"), ("nullable", 1)):
        invalid = json.loads(json.dumps(document))
        invalid["value"][field] = wrong
        assert list(validator.iter_errors(invalid))
    del document["value"]["defaulted"]
    assert list(validator.iter_errors(document))


@dataclass(frozen=True)
class _UnsupportedRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/test/unsupported-configuration"
    operand: float


def test_unknown_annotations_fail_closed_without_any_schema_fallback():
    with pytest.raises(TypeError, match="unsupported configuration schema annotation"):
        project_record(_UnsupportedRecord)


def test_import_schemas_preserve_extensible_metadata_and_cli_resource_export():
    consumer = next(item for item in CONSUMERS if item.consumer == "finite-prior-census")
    schema = configuration_schema(consumer.schema_id)
    document = json.loads((ROOT / consumer.supplied_inputs[0]).read_bytes())
    document["historical_measurement"] = 0.25
    _offline(schema).validate(document)
    result = CliRunner().invoke(config_app, ["schema", "--schema-id", consumer.schema_id])
    assert result.exit_code == 0 and result.stderr == ""
    assert json.loads(result.stdout) == schema
    unknown = CliRunner().invoke(config_app, ["schema", "--schema-id", "unknown/root"])
    assert unknown.exit_code == 2 and unknown.stdout == "" and unknown.stderr
