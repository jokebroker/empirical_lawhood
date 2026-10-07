"""Configuration authoring preserves consumers, exact operands and custody limits."""

from dataclasses import replace
from decimal import Decimal, localcontext
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest
from typer.testing import CliRunner

from empirical_lawhood.api import configuration
from empirical_lawhood.api.configuration import ConfigurationError, prepare_configuration, validate_configuration
from empirical_lawhood.api.configuration_registry import CONSUMERS, consumer_by_id
from empirical_lawhood.cli.configuration import config_app
from empirical_lawhood.kernel.decoding import decode_canonical_bytes

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "experiments/rc-ladder-response/model.json"
BRIAN = ROOT / "experiments/neuron-current-response/config.json"
SUPPLIED = tuple((consumer, path) for consumer in CONSUMERS for path in consumer.supplied_inputs if path.endswith(".json"))


def _copy(tmp_path, document, name="input.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path


def test_registry_exactly_accounts_for_public_inputs_without_widening_roots():
    actual = {path.relative_to(ROOT).as_posix() for path in (ROOT / "experiments").glob("*/*.json")}
    actual.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "experiments").glob("*/exposed-inputs/*.json"))
    registered = {path for consumer in CONSUMERS for path in consumer.supplied_inputs if path.startswith("experiments/")}
    assert actual == registered
    assert {path for consumer in CONSUMERS for path in consumer.supplied_inputs if path.startswith("configs/")} == {
        "configs/operator-storage.example.json", "configs/sources/fair-mast-level2.json",
        "configs/templates/evidence-provenance.yaml",
    }
    assert len({consumer.consumer for consumer in CONSUMERS}) == len(CONSUMERS)
    assert {consumer.disposition for consumer in CONSUMERS} == {"editable", "retained", "import", "helper"}


@pytest.mark.parametrize("consumer,path", SUPPLIED, ids=[path for _, path in SUPPLIED])
def test_supplied_values_validate_without_network_or_provider_contact(consumer, path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("configuration validation contacted a network")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    report = validate_configuration(ROOT / path, consumer=consumer.consumer)
    assert report["status"] == "VALIDATED"
    assert report["input_sha256"] == sha256((ROOT / path).read_bytes()).hexdigest()
    assert report["external_prerequisites_checked"] is False
    assert report["authority_granted"] is False
    assert report["preparation_allowed"] == (consumer.disposition == "editable")
    assert report["consumer_byte_ready"] is True


@pytest.mark.parametrize("original", [RC, BRIAN], ids=["rc", "brian2"])
def test_pretty_copy_round_trip_reproduces_exact_existing_bytes(original, tmp_path):
    source = _copy(tmp_path, json.loads(original.read_bytes()))
    before = source.read_bytes()
    report = validate_configuration(source)
    assert report["canonical_byte_ready"] is False
    assert report["consumer_byte_ready"] is False
    assert report["preparation_required"] is True
    output = tmp_path / "canonical.json"
    prepared = prepare_configuration(source, output)
    assert source.read_bytes() == before
    assert output.read_bytes() == original.read_bytes()
    assert prepared["output_sha256"] == sha256(original.read_bytes()).hexdigest()
    assert prepared["input_sha256"] == sha256(before).hexdigest()
    assert validate_configuration(output)["canonical_byte_ready"] is True


def test_changed_rc_decimal_is_lossless_under_small_arithmetic_context(tmp_path):
    document = json.loads(RC.read_bytes())
    text = "0.00123456789012345678901234567890123456789"
    document["value"]["capacitances_farads"][0] = {"decimal": text}
    source = _copy(tmp_path, document)
    with localcontext() as context:
        context.prec = 6
        before = context.copy()
        output = tmp_path / "canonical.json"
        prepare_configuration(source, output, consumer="rc-ladder-model")
        consumer = consumer_by_id("rc-ladder-model")
        record = decode_canonical_bytes(output.read_bytes(), consumer.record_type(), maximum_bytes=consumer.maximum_bytes)
        assert record.capacitances_farads[0].as_tuple() == Decimal(text).as_tuple()
        assert context.prec == before.prec and context.rounding == before.rounding
        assert context.traps == before.traps
    assert output.read_bytes() != RC.read_bytes()


@pytest.mark.parametrize("mutation,code,fragment", [
    (lambda d: d.update(version="9.0.0"), "WRONG_REVISION", "version"),
    (lambda d: d.update(extra=True), "OWNER_VALIDATION_REFUSED", "envelope fields"),
    (lambda d: d["value"].update(extra=True), "OWNER_VALIDATION_REFUSED", "fields differ"),
    (lambda d: d["value"].update(scale_cells=True), "OWNER_VALIDATION_REFUSED", "integer"),
    (lambda d: d["value"]["action_segments"][0].update(schema="unknown/nested"), "OWNER_VALIDATION_REFUSED", "schema"),
    (lambda d: d["value"]["action_segments"][0].update(version="9.0.0"), "OWNER_VALIDATION_REFUSED", "version"),
    (lambda d: d["value"]["left_source_resistance_ohms"].update(decimal="NaN"), "INVALID_DECIMAL", "finite"),
    (lambda d: d["value"]["left_source_resistance_ohms"].update(decimal="invalid"), "INVALID_DECIMAL", "Decimal"),
    (lambda d: d["value"].update(capacitances_farads=[]), "SEMANTIC_REFUSAL", "capacitance"),
    (lambda d: d["value"]["action_segments"][0]["value"]["end_seconds"].update(decimal="0"), "SEMANTIC_REFUSAL", "interval"),
])
def test_structural_and_semantic_refusals_retain_owner_diagnostics(mutation, code, fragment, tmp_path):
    document = json.loads(RC.read_bytes())
    mutation(document)
    source = _copy(tmp_path, document)
    with pytest.raises(ConfigurationError) as caught:
        prepare_configuration(source, tmp_path / "output.json")
    assert caught.value.code == code
    assert fragment.lower() in str(caught.value).lower()
    assert not (tmp_path / "output.json").exists()


@pytest.mark.parametrize("content", [
    b'{"schema":"a","schema":"b"}',
    b'{"schema":"unknown/schema","value":NaN}',
    b'{"schema":',
])
def test_ambiguous_and_invalid_syntax_is_rejected(content, tmp_path):
    source = tmp_path / "input.json"
    source.write_bytes(content)
    with pytest.raises(ConfigurationError) as caught:
        validate_configuration(source)
    assert caught.value.code == "INVALID_SYNTAX"


def test_bare_fractional_json_number_is_not_coerced(tmp_path):
    document = json.loads(RC.read_bytes())
    document["value"]["scale_cells"] = 4.0
    with pytest.raises(ConfigurationError, match="binary numbers"):
        validate_configuration(_copy(tmp_path, document))


def test_unknown_and_authority_roots_have_no_fallback_parser(tmp_path):
    for schema in ("unregistered/module.Class", "empirical-lawhood/planning/study-operation-authority"):
        with pytest.raises(ConfigurationError) as caught:
            validate_configuration(_copy(tmp_path, {"schema": schema, "version": "1.0.0", "value": {}}))
        assert caught.value.code == "UNKNOWN_SCHEMA"


def test_consumer_choice_is_explicit_when_ambiguous_and_must_match_root(tmp_path, monkeypatch):
    selected = consumer_by_id("rc-ladder-model")
    monkeypatch.setattr(configuration, "consumers_for_schema", lambda _schema: (selected, replace(selected, consumer="other-role")))
    with pytest.raises(ConfigurationError) as caught:
        validate_configuration(RC)
    assert caught.value.code == "CONSUMER_REQUIRED"
    assert validate_configuration(RC, consumer="rc-ladder-model")["consumer"] == "rc-ladder-model"
    with pytest.raises(ConfigurationError) as caught:
        validate_configuration(RC, consumer="brian2-current")
    assert caught.value.code == "CONSUMER_SCHEMA_MISMATCH"


@pytest.mark.parametrize("consumer", [item for item in CONSUMERS if item.disposition in {"retained", "import", "helper"} and item.schema_id and item.supplied_inputs])
def test_retained_import_and_helper_inputs_cannot_be_prepared(consumer, tmp_path):
    source = ROOT / consumer.supplied_inputs[0]
    before = source.read_bytes()
    with pytest.raises(ConfigurationError) as caught:
        prepare_configuration(source, tmp_path / "output.json", consumer=consumer.consumer)
    assert caught.value.code == "PREPARATION_FORBIDDEN"
    assert source.read_bytes() == before
    assert not (tmp_path / "output.json").exists()


def test_pretty_retained_profile_cannot_repair_byte_custody(tmp_path):
    consumer = consumer_by_id("reactor-assigned-profile")
    path = _copy(tmp_path, json.loads((ROOT / consumer.supplied_inputs[0]).read_bytes()))
    with pytest.raises(ConfigurationError) as caught:
        validate_configuration(path)
    assert caught.value.code == "RETAINED_BYTES_REFUSED"


def test_import_extensions_keep_owning_parser_policy_and_raw_identity(tmp_path):
    consumer = consumer_by_id("finite-prior-census")
    document = json.loads((ROOT / consumer.supplied_inputs[0]).read_bytes())
    document["historical_extra_measurement"] = 0.25
    path = _copy(tmp_path, document)
    report = validate_configuration(path)
    assert report["canonical_sha256"] is None
    assert report["input_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert report["validation_scope"] == "owning census import only"
    assert report["external_prerequisites_checked"] is False


def test_provenance_template_has_explicit_helper_only_disposition():
    consumer = consumer_by_id("evidence-provenance-template")
    assert consumer.maximum_bytes is None and consumer.authoring_maximum_bytes is None
    with pytest.raises(ConfigurationError) as caught:
        validate_configuration(ROOT / "configs/templates/evidence-provenance.yaml", consumer="evidence-provenance-template")
    assert caught.value.code == "HELPER_ONLY"


def test_declared_yaml_authoring_round_trip_and_exact_json_consumer_refusal(tmp_path):
    import yaml
    source = ROOT / "experiments/battery-response-and-restart/config.json"
    path = tmp_path / "editable.yaml"
    path.write_text(yaml.safe_dump(json.loads(source.read_bytes())))
    output = tmp_path / "canonical.json"
    prepare_configuration(path, output)
    assert output.read_bytes() == source.read_bytes()
    path.write_text(yaml.safe_dump(json.loads(RC.read_bytes())))
    with pytest.raises(ConfigurationError) as caught:
        validate_configuration(path)
    assert caught.value.code == "UNSUPPORTED_FORMAT"
    path.write_text("schema: &alias example\nvalue: *alias\n")
    with pytest.raises(ConfigurationError, match="aliases, anchors"):
        validate_configuration(path)


def test_authoring_and_prepared_byte_bounds_are_separate(tmp_path):
    consumer = consumer_by_id("rc-ladder-model")
    source = tmp_path / "large.json"
    source.write_bytes(RC.read_bytes() + b" " * consumer.maximum_bytes)
    report = validate_configuration(source, consumer=consumer.consumer)
    assert not report["consumer_byte_ready"]
    assert report["preparation_required"]
    output = tmp_path / "prepared.json"
    prepare_configuration(source, output, consumer=consumer.consumer)
    assert output.read_bytes() == RC.read_bytes()
    source.write_bytes(RC.read_bytes() + b" " * consumer.authoring_maximum_bytes)
    with pytest.raises(ConfigurationError) as caught:
        validate_configuration(source, consumer=consumer.consumer)
    assert caught.value.code == "INPUT_READ_REFUSED"


def test_short_decimal_exponent_and_aggregate_expansion_refuse_before_constructor(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized config reached a constructor")
    monkeypatch.setattr(configuration, "decode_record", forbidden)
    document = json.loads(RC.read_bytes())
    document["value"]["left_source_resistance_ohms"] = {"decimal": "1e+999999999"}
    with pytest.raises(ConfigurationError) as caught:
        validate_configuration(_copy(tmp_path, document))
    assert caught.value.code == "CONSUMER_BYTE_LIMIT"
    document = json.loads(RC.read_bytes())
    document["value"]["initial_voltages_volts"] = [{"decimal": "1e+20000"}] * 4
    with pytest.raises(ConfigurationError) as caught:
        validate_configuration(_copy(tmp_path, document))
    assert caught.value.code == "CONSUMER_BYTE_LIMIT"


def test_json_escape_expansion_refuses_before_constructor_without_serializing(tmp_path, monkeypatch):
    document = json.loads(RC.read_bytes())
    document["value"]["config_id"] = "\u00e9" * 12000
    source = tmp_path / "unicode.json"
    source.write_text(json.dumps(document, ensure_ascii=False))
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized escaped output reached constructor")
    monkeypatch.setattr(configuration, "decode_record", forbidden)
    with pytest.raises(ConfigurationError) as caught:
        validate_configuration(source)
    assert caught.value.code == "CONSUMER_BYTE_LIMIT"


@pytest.mark.parametrize("text", ["plain", '"\\\b\f\n\r\t', "\x00\x01\x1f\x7f", "\u00e9\u2603\U0001f680"])
def test_json_string_escape_size_matches_independent_encoder(text):
    assert configuration._string_size(text) == len(json.dumps(text, ensure_ascii=True))


def test_prepare_refuses_existing_files_in_place_and_symlinks(tmp_path):
    source = _copy(tmp_path, json.loads(RC.read_bytes()))
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"preserve me")
    link = tmp_path / "linked.json"
    link.symlink_to(existing)
    directory_link = tmp_path / "linked-parent"
    directory_link.symlink_to(tmp_path, target_is_directory=True)
    for output in (source, existing, link, directory_link / "new.json"):
        with pytest.raises(ConfigurationError) as caught:
            prepare_configuration(source, output)
        assert caught.value.code == "OUTPUT_REFUSED"
    assert existing.read_bytes() == b"preserve me"
    assert not (tmp_path / "new.json").exists()
    source_link = tmp_path / "source-link.json"
    source_link.symlink_to(source)
    with pytest.raises(ConfigurationError) as caught:
        validate_configuration(source_link)
    assert caught.value.code == "INPUT_READ_REFUSED"


@pytest.mark.parametrize("failure", [OSError("disk error"), KeyboardInterrupt(), SystemExit(7)])
def test_failed_new_output_is_removed_and_original_is_preserved(tmp_path, monkeypatch, failure):
    source = _copy(tmp_path, json.loads(RC.read_bytes()))
    before = source.read_bytes()
    output = tmp_path / "prepared.json"
    def fail(_descriptor):
        raise failure
    monkeypatch.setattr(configuration.os, "fsync", fail)
    with pytest.raises(ConfigurationError if isinstance(failure, OSError) else type(failure)):
        prepare_configuration(source, output)
    assert not output.exists()
    assert source.read_bytes() == before


def test_cli_json_success_stdout_refusal_stderr_and_click_exit_contract(tmp_path):
    runner = CliRunner()
    success = runner.invoke(config_app, ["validate", "--config", str(RC), "--format", "json"])
    assert success.exit_code == 0
    assert json.loads(success.stdout)["status"] == "VALIDATED"
    assert success.stderr == ""
    failure = runner.invoke(config_app, ["validate", "--config", str(tmp_path / "missing.json"), "--format", "json"])
    assert failure.exit_code == 2
    assert failure.stdout == ""
    assert json.loads(failure.stderr)["code"] == "INPUT_READ_REFUSED"
    usage = runner.invoke(config_app, ["validate", "--format", "json"])
    assert usage.exit_code == 2 and usage.stdout == "" and usage.stderr
