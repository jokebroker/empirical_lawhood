# SPDX-License-Identifier: MPL-2.0
"""Public RC copy/edit/prepare/result journey without simulator or authority."""

from fractions import Fraction
import json
from pathlib import Path

from typer.testing import CliRunner

from empirical_lawhood.cli.app import app

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'experiments/rc-information/input.json'


def test_public_edited_input_requires_prepare_and_retains_scaled_result(tmp_path):
    runner = CliRunner()
    baseline = runner.invoke(app, ['example', 'rc-information', '--config', str(BASELINE)])
    assert baseline.exit_code == 0, baseline.output
    original = json.loads(baseline.stdout)['value']['result']['value']
    editable = json.loads(BASELINE.read_bytes())
    editable['value']['voltage_scale_volts'] = {'decimal': '2'}
    edited = tmp_path / 'edited.json'
    edited.write_text(json.dumps(editable, indent=2))
    refused = runner.invoke(app, ['example', 'rc-information', '--config', str(edited)])
    assert refused.exit_code == 2
    assert 'prepare' in refused.output.lower()
    prepared = tmp_path / 'prepared.json'
    result = runner.invoke(app, ['config', 'prepare', '--config', str(edited), '--consumer', 'rc-information', '--output-file', str(prepared)])
    assert result.exit_code == 0, result.output
    destination = tmp_path / 'rc-attempt'
    result = runner.invoke(app, ['example', 'rc-information', '--config', str(prepared), '--output-dir', str(destination)])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    changed = report['value']['result']['value']
    key = 'identical_input_minimax_lower_bound_approx_volts'
    assert Fraction(changed[key]['decimal']) == Fraction(original[key]['decimal']) * 2
    saved = json.loads((destination / 'report.json').read_bytes())
    assert saved == report
    attempt = json.loads((destination / 'invocation.json').read_bytes())
    assert attempt['native_contact'] == 'none'
    assert attempt['scientific_adjudication'] == 'NOT_PERFORMED'
    assert edited.read_text() == json.dumps(editable, indent=2)


def test_wrong_common_input_refuses_before_analytical_calculation(tmp_path, monkeypatch):
    from empirical_lawhood.api import rc_information
    def forbidden(_config):
        raise AssertionError('invalid common-input premise reached calculation')
    monkeypatch.setattr(rc_information, 'calculate_rc_information', forbidden)
    document = json.loads(BASELINE.read_bytes())
    document['value']['predictor_input_b_sha256'] = '0' * 64
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(document))
    result = CliRunner().invoke(app, ['example', 'rc-information', '--config', str(bad)])
    assert result.exit_code == 2
    assert 'same predictor input' in result.output


def test_public_discovery_names_actual_rc_operation():
    result = CliRunner().invoke(app, ['workflow', 'show', 'rc-information', '--format', 'json'])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report['prerequisites_inspected'] is False
    assert report['operations'][0]['command'] == 'example rc-information'
    assert report['operations'][0]['invocation']['storage'] == 'none'
