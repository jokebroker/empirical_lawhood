# SPDX-License-Identifier: MPL-2.0
"""Smoke the selected wheel through its real entry point, outside the checkout."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    checkout, output_dir = args.checkout.resolve(), args.output_dir.resolve()
    assert not Path.cwd().is_relative_to(checkout)
    import empirical_lawhood
    assert not Path(empirical_lawhood.__file__).resolve().is_relative_to(checkout)
    selected_version = tomllib.loads((checkout / "pyproject.toml").read_text())["project"]["version"]
    assert version("empirical-lawhood") == empirical_lawhood.__version__ == selected_version
    from empirical_lawhood.cli.entry import (
        NUMERICAL_THREAD_ENV_VARS, NUMERICAL_BOOTSTRAP_ENV_VAR, enforce_single_thread_environment,
    )
    enforce_single_thread_environment()
    from typer.main import get_command
    from empirical_lawhood.cli.app import app
    from empirical_lawhood.cli.introspection import canonical_click_tree
    from empirical_lawhood.api import load_authoring
    from empirical_lawhood.api.configuration_registry import consumer_by_id

    cli = str(Path(sys.executable).with_name("empirical-lawhood"))
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    def invoke(arguments, *, code=0, env=environment):
        result = subprocess.run([cli, *arguments], cwd=output_dir, env=env,
                                capture_output=True, text=True, timeout=120)
        if result.returncode != code:
            raise AssertionError((arguments, result.returncode, result.stdout, result.stderr))
        return result

    invoke(["--help"])
    leaves = [fact.path for fact in canonical_click_tree(get_command(app)) if fact.command_kind == "command"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        # Each process runs the actual installed script, including the bootstrap.
        list(pool.map(lambda path: invoke([*path, "--help"]), leaves))
    print(f"installed leaf help: {len(leaves)} passed", flush=True)

    trace_dir = output_dir / "bootstrap-trace"
    trace_dir.mkdir()
    observed = output_dir / "bootstrap.json"
    trace_source = (
        "import json, os, sys\n"
        "class Observer:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'numpy':\n"
        f"            with open({str(observed)!r}, 'w') as f:\n"
        f"                json.dump({{k:os.environ.get(k) for k in {(*NUMERICAL_THREAD_ENV_VARS, NUMERICAL_BOOTSTRAP_ENV_VAR)!r}}}, f)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Observer())\n"
    )
    (trace_dir / "sitecustomize.py").write_text(trace_source)
    conflicting = dict(environment, PYTHONPATH=str(trace_dir))
    conflicting.update({name: "64" for name in NUMERICAL_THREAD_ENV_VARS})
    conflicting.pop(NUMERICAL_BOOTSTRAP_ENV_VAR, None)
    doctor = json.loads(invoke(["doctor", "--route", "reactor", "--format", "json"], env=conflicting).stdout)
    assert doctor["payload"]["repository_root"] == ""
    assert all(row["version_matches"] for row in doctor["payload"]["optional_dependencies"])
    bootstrap = json.loads(observed.read_text())
    assert all(bootstrap[name] == "1" for name in NUMERICAL_THREAD_ENV_VARS)
    assert bootstrap[NUMERICAL_BOOTSTRAP_ENV_VAR] == "single-thread-before-numerical-import"

    capabilities = json.loads(invoke(["capability", "list", "--limit", "5", "--format", "json"]).stdout)
    assert capabilities["status"] == "SUCCEEDED"
    # Installed static discovery and schemas must work away from any checkout,
    # without optional simulator imports or a project/storage configuration.
    existing = set(output_dir.iterdir())
    workflows = json.loads(invoke(["workflow", "list", "--format", "json"]).stdout)
    assert {"rc-information", "rc-challenges", "matrix-inputs", "finite-response-law", "preparation-applicability", "matrix-geometry", "selected-events", "matrix-history-analysis", "matrix-tangent", "matrix-transient", "preparation-diagnostics"} <= {item["workflow_id"] for item in workflows["workflows"]}
    rc_workflow = json.loads(invoke(["workflow", "show", "rc-ladder-response", "--format", "json"]).stdout)
    assert rc_workflow["prerequisites_inspected"] is False
    model_path = checkout / "experiments/rc-ladder-response/model.json"
    model = model_path.read_bytes()
    schema_id = json.loads(model)["schema"]
    schema = json.loads(invoke(["config", "schema", "--schema-id", schema_id]).stdout)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$defs"] and schema["x-consumer"] == "rc-ladder-model"
    readiness = json.loads(invoke(["config", "validate", "--config", str(model_path), "--format", "json"]).stdout)
    assert readiness["canonical_byte_ready"] is True
    assert set(output_dir.iterdir()) == existing
    pretty = output_dir / "pretty-model.json"
    pretty.write_text(json.dumps(json.loads(model), indent=2))
    prepared = output_dir / "prepared-model.json"
    invoke(["config", "prepare", "--config", str(pretty), "--output-file", str(prepared), "--format", "json"])
    assert prepared.read_bytes() == model and model_path.read_bytes() == model
    analytical = json.loads(invoke(["example", "rc-information", "--config", str(checkout / "experiments/rc-information/input.json")]).stdout)
    assert analytical["schema"] == "empirical-lawhood/api/rc-information-report"
    lower_dir = output_dir / "original-f"
    lower = json.loads(invoke(["campaign", "original-f-export", "--output-dir", str(lower_dir), "--payload-id", "wheel.original-f"]).stdout)
    checked = json.loads(invoke(["campaign", "original-f-check", "--operand", str(lower_dir / "original-f.canonical.json"), "--source-directory", str(lower_dir)]).stdout)
    assert lower["identity"] == checked["identity"] and checked["qualification_performed"] is False
    nomination_dir = output_dir / "nomination"
    nominated = json.loads(invoke(["campaign", "nomination-export", "--output-dir", str(nomination_dir), "--nomination-id", "wheel.nomination"]).stdout)
    authenticated = json.loads(invoke(["campaign", "nomination-check", "--operand", str(nomination_dir / "nomination.canonical.json"), "--source-directory", str(nomination_dir)]).stdout)
    assert nominated["identity"] == authenticated["identity"] and authenticated["refitting_performed"] is False
    geometry = json.loads(invoke(["campaign", "matrix-geometry-prove", "--input", str(checkout / "experiments/matrix-geometry/input.json")]).stdout)
    assert geometry["scientific_execution_performed"] is False and geometry["potential_task_count"] == 17118
    for consumer in ("matrix-geometry", "selected-parent", "matrix-history-allocation", "matrix-history-source", "matrix-history-analysis", "matrix-tangent", "matrix-transient-analysis", "matrix-baseline-analysis", "preparation-diagnostics"):
        schema_id = consumer_by_id(consumer).schema_id
        schema = json.loads(invoke(["config", "schema", "--schema-id", schema_id]).stdout)
        assert schema["x-consumer"] == consumer
    history = output_dir / "history-allocation.json"
    allocated = json.loads(invoke(["campaign", "matrix-history-allocation", "--allocation-id", "wheel.history", "--namespace", "wheel.history", "--master-seed", "745121", "--output", str(history)]).stdout)
    assert allocated["roots"] == 256 and allocated["eligibility_granted"] is False
    validation = json.loads(invoke(["config", "validate", "--consumer", "matrix-history-allocation", "--config", str(history), "--format", "json"]).stdout)
    assert validation["canonical_byte_ready"] is True
    # Missing config is retained by the actual installed early CLI boundary.
    early = output_dir / "early-refusal"
    invoke(["campaign", "rc-ladder-native-check", "--config", str(output_dir / "missing.json"),
            "--output-dir", str(early)], code=2)
    early_record = json.loads((early / "invocation.json").read_bytes())
    assert early_record["exit_code"] == 2 and early_record["operation_completed"] is False
    assert (early / "failure.json").is_file() and not (early / "report.json").exists()
    power = json.loads(invoke(["campaign", "response-composition-power"]).stdout)
    assert (power["n"], power["required_successes"], power["maximum_false_admissions"]) == (64, 52, 2)
    assert len(power["multinomial_cases"]) == 12 and len(power["paired_use_sensitivity"]) == 5
    assert power["native_contact"] is False and power["outcome_reads"] is False
    assert power["campaign_issued"] is False
    package = load_authoring(checkout / "tests/fixtures/reference-campaign.json")
    spec = output_dir / "system.json"
    spec.write_bytes(package.system.canonical_bytes())
    assert json.loads(invoke(["system", "validate", "--spec", str(spec), "--format", "json"]).stdout)["status"] == "SUCCEEDED"
    source_refusal = invoke(["--project-root", str(checkout), "db", "check", "--format", "json"], code=2)
    assert "executing" in source_refusal.stderr

    for command, relative, reason in (
        ("reactor-batch-input-check", "experiments/reactor-response/batch-input.json", "SOURCE_ROOT_REQUIRED"),
        ('dependent-response-input-check', "experiments/prepared-response/prepared-dependent-refinement-input.json", "SOURCE_ROOT_REQUIRED"),
        ('finite-response-input-check', "experiments/finite-response-law/finite-calibration-input.json", "SOURCE_ROOT_REQUIRED"),
    ):
        result = invoke(["campaign", command, "--config", str(checkout / relative)], code=3)
        assert reason in result.stderr, result.stderr
    canary = json.loads(invoke(["campaign", 'prepared-response-native-check', "--config",
                               str(checkout / "experiments/prepared-response/prepared-canary.json")]).stdout)
    assert canary["campaign_issued"] is False

    demo = output_dir / "reactor-prefix"
    invoke(["example", "reactor-prefix", "--output-dir", str(demo)])
    report = json.loads((demo / "report.json").read_text())
    assert (report["complete_branches"], report["observed_deliveries"], report["independent_units"]) == (20, 40, 5)
    original = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in demo.iterdir()}
    invoke(["example", "reactor-prefix", "--output-dir", str(demo)], code=1)
    assert original == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in demo.iterdir()}

    # Retained partial output is verified from this installed wheel as well.
    from empirical_lawhood.examples import reactor_prefix
    delegate = reactor_prefix.acquire_prefix_branch
    calls = 0

    def stop_after_one(*values):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption for packaging conformance")
        return delegate(*values)

    reactor_prefix.acquire_prefix_branch = stop_after_one
    partial_dir = output_dir / "partial-prefix"
    try:
        reactor_prefix.run_reactor_prefix(partial_dir)
    except reactor_prefix.ReactorPrefixRunFailed:
        pass
    else:
        raise AssertionError("partial prefix must fail")
    finally:
        reactor_prefix.acquire_prefix_branch = delegate
    partial = json.loads((partial_dir / "partial.json").read_text())
    assert (partial["completed_branches"], partial["observed_deliveries"]) == (1, 2)
    assert not (partial_dir / "report.json").exists() and not (partial_dir / "panel.json").exists()
    (output_dir / "smoke-result.json").write_text(json.dumps({
        "status": "PASSED", "installed_module": empirical_lawhood.__file__,
        "leaf_help_count": len(leaves), "bootstrap_before_numpy": bootstrap,
        "complete_branches": 20, "independent_units": 5, "partial_branches": 1,
        "evidence_ceiling": "exposed demonstration; no scientific qualification",
        "workflow_count": len(workflows["workflows"]), "offline_schema_and_preparation": True,
        "current_analytical_and_frozen_operand_delivery": True,
        "retained_early_refusal": True,
    }, indent=2) + "\n")
    print("installed wheel smoke passed", flush=True)


if __name__ == "__main__":
    main()
