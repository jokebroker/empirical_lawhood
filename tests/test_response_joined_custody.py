# SPDX-License-Identifier: MPL-2.0
"""Public selectors join real custody replay to providers, with no reader stubs."""

import base64
from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import numpy as np
import pytest
from typer.testing import CliRunner

from tests.response_custody_fixtures import publish_parent
from tests.response_dependent_fixtures import _selection_config, calibration_records as calibration_records, development_reports as development_reports, qualification_fixture as qualification_fixture
from tests.test_response_dependent_binding import _authoring_input
from tests.test_doctor_profile import _profile
from tests.test_finite_response_retained_development_input import fixture as finite_retained_development_parent
from empirical_lawhood.adapters.composition.finite_response_law.informative_composition_input import FiniteResponseLawInformativeCompositionParent
from empirical_lawhood.adapters.composition.response_composition.scalar_parent import ResponseCompositionScalarParent, SHAPES
from empirical_lawhood.adapters.methods.prepared_response.qualification import evaluate_prepared_response_source_qualification
from empirical_lawhood.adapters.methods.response_composition.development import ResponseCompositionDevelopmentSpec
from empirical_lawhood.cli.app import app

ROOT = Path(__file__).resolve().parents[1]


def _cli_args(profile_path, route, args):
    command = 'response-composition-input-check' if route == "response-composition" else 'dependent-response-input-check'
    config = 'causal-transfer-audit/analysis-input.json' if route == "response-composition" else f"prepared-response/prepared-{route}-input.json"
    invocation = ["--project-root", str(ROOT), "--operator-profile", str(profile_path),
                  "campaign", command, "--config", str(ROOT / "experiments" / config)]
    for key, value in args.items():
        if key != "authority_store":
            invocation.extend(["--" + key.replace("_", "-"), str(value)])
    return invocation


@pytest.mark.parametrize("route", ("dependent-refinement", "fresh-response-calibration"))
def test_real_public_dependent_join_and_tamper(
    tmp_path, route, qualification_fixture, development_reports, calibration_records,
):
    if route == "dependent-refinement":
        config, members = qualification_fixture
        parent = evaluate_prepared_response_source_qualification(config, members)
        native = config.projection.native_spec
        costs = _selection_config(development_reports[0]).candidate_costs
    else:
        records, _, members = calibration_records
        native, policy, _, _ = records
        parent = policy.development_library
        costs = ()
    roots = tuple(sorted(r.root_id for r in (
        config.projection.native_spec.roots if route == "dependent-refinement"
        else parent.config.fit.projection.native_spec.roots
    )))
    with TemporaryDirectory(prefix="prepared-response-joined-", dir="/dev/shm") as directory:
        target = Path(directory)
        (target / "artifacts").mkdir()
        profile = _profile(target)
        args, parent_path = publish_parent(target, profile=profile, repo=ROOT,
            route=route, parent=parent, roots=roots, members=members)
        args["authoring_input"] = _authoring_input(args["source_root"], route, native, roots, costs)
        profile_path = tmp_path / "profile.json"
        profile_path.write_bytes(profile.canonical_bytes())
        invocation = _cli_args(profile_path, route, args)
        before = {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}
        result = CliRunner().invoke(app, invocation)
        assert result.exit_code == 0, result.output
        report = json.loads(result.stdout)
        assert report["provider_built"] and report["parent_custody_authenticated"]
        assert report["independent_units"] == (128 if route == "dependent-refinement" else 192)
        assert report["factory"] == ('PreparedResponseDevelopmentSourceFactory' if route == "dependent-refinement" else 'PreparedResponseCalibrationSourceFactory')
        assert report["native_tasks_executed"] == 0
        assert not report["prospective_issue_eligible"] and not report["campaign_issued"]
        assert {p: p.read_bytes() for p in target.rglob("*") if p.is_file()} == before
        # A valid JSON mutation is rejected by custody before the scientific decoder.
        parent_path.write_bytes(parent_path.read_bytes() + b" ")
        refusal = CliRunner().invoke(app, invocation)
        assert refusal.exit_code == 3 and refusal.stdout == ""
        assert "RESPONSE_PARENT_IDENTITY_BYTES_MISMATCH" in refusal.stderr


def test_real_public_response_composition_join_and_durable_grant_tamper(tmp_path):
    roots = tuple(f"synthetic.qualification.{c}.r{i:03d}" for c in ("assembling", "prepared") for i in range(16))
    parent = ResponseCompositionScalarParent(roots, ResponseCompositionDevelopmentSpec(*("a" * 64 for _ in range(4))),
        tuple((key, base64.b64encode(np.zeros(shape, dtype="<f8").tobytes()).decode())
              for key, shape in SHAPES.items()))
    with TemporaryDirectory(prefix="prepared-response-joined-", dir="/dev/shm") as directory:
        target = Path(directory)
        (target / "artifacts").mkdir()
        profile = _profile(target)
        args, _ = publish_parent(target, profile=profile, repo=ROOT,
                                 route="response-composition", parent=parent, roots=roots)
        profile_path = tmp_path / "profile.json"
        profile_path.write_bytes(profile.canonical_bytes())
        invocation = _cli_args(profile_path, "response-composition", args)
        result = CliRunner().invoke(app, invocation)
        assert result.exit_code == 0, result.output
        report = json.loads(result.stdout)
        assert report["parent_custody_authenticated"] and report["independent_units"] == 32
        assert report["consumer_selected"].endswith("develop_context")
        assert report["analysis_tasks_executed"] == 0 and not report["prospective_issue_eligible"]
        args["reveal_record"].write_bytes(b"{}")
        refusal = CliRunner().invoke(app, invocation)
        assert refusal.exit_code == 3 and refusal.stdout == ""
        assert "Response-composition input refused" in refusal.stderr and "refused" in refusal.stderr


def _finite_cli_args(profile_path, stage, args, *, plan, prior, assignment=None, authoring_input=None):
    files = {
        "informative-composition": 'informative-composition-input.json',
        "prospective-evaluation": 'prospective-evaluation-input.json',
        "prospective-continuation": 'prospective-continuation-input.json',
        "preparation-screening": 'preparation-screen-input.json',
    }
    invocation = ["--project-root", str(ROOT), "--operator-profile", str(profile_path),
                  "campaign", 'finite-response-stage-input-check', "--config",
                  str(ROOT / "experiments/finite-response-law" / files[stage]),
                  "--plan", str(plan), "--prior-exposure", str(prior)]
    for key, value in args.items():
        if key != "authority_store" and value is not None:
            invocation.extend(["--" + key.replace("_", "-"), str(value)])
    for key, value in (("assignment", assignment), ("authoring-input", authoring_input)):
        if value is not None:
            invocation.extend(["--" + key, str(value)])
    return invocation


def _finite_inputs(source_root, roots):
    plan = source_root / "plan.md"
    plan.write_bytes((ROOT / 'src/empirical_lawhood/adapters/methods/finite_response_law/specification.md').read_bytes())
    prior = source_root / "prior.json"
    prior.write_text(json.dumps({
        "schema": 'empirical-lawhood/methods/finite-response-law/native-exposure-metadata',
        "excluded_unit_ids": list(roots), "proposed_unit_ids": [],
        "excluded_seed_ids": ["synthetic.previous.stream"], "proposed_seed_ids": [],
    }))
    return plan, prior


def _snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_real_public_finite_retained_method_join_and_tamper(tmp_path):
    parent = finite_retained_development_parent()
    roots = tuple(sorted(r for _, r in parent.root_identity_map))
    with TemporaryDirectory(prefix='finite-response-parent-joined-', dir="/dev/shm") as directory:
        target = Path(directory)
        (target / "artifacts").mkdir()
        profile = _profile(target)
        args, parent_path = publish_parent(target, profile=profile, repo=ROOT,
                                          route="finite-response-law-informative-composition", parent=parent, roots=roots)
        plan, prior = _finite_inputs(args["source_root"], roots)
        profile_path = tmp_path / "profile.json"
        profile_path.write_bytes(profile.canonical_bytes())
        invocation = _finite_cli_args(profile_path, "informative-composition", args, plan=plan, prior=prior)
        before = _snapshot(target)
        result = CliRunner().invoke(app, invocation)
        assert result.exit_code == 0, result.output
        report = json.loads(result.stdout)
        assert report["consumer_selected"] is True and report["provider_built"] is False
        assert report["binding_selected"] == "finite_response_law.development.develop_fold+summarize"
        assert report["parent_custody_authenticated"] and report["target_reveal_authorized"]
        assert report["target_analysis_authorized"]
        assert (report["independent_units"], report["complete_decision_roots"]) == (48, 16)
        assert (report["outer_folds"], report["inner_folds"], report["nested_views_per_unit"]) == (4, 3, 2)
        assert report["native_tasks_executed"] == report["analysis_tasks_executed"] == 0
        assert report["campaign_candidate_compiled"] is False and report["campaign_issued"] is False
        assert report["evidence_role"] == "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
        assert _snapshot(target) == before
        parent_path.write_bytes(parent_path.read_bytes() + b" ")
        refused = CliRunner().invoke(app, invocation)
        assert refused.exit_code == 3 and refused.stdout == ""
        assert "RESPONSE_PARENT_IDENTITY_BYTES_MISMATCH" in refused.stderr


@pytest.mark.parametrize(("missing", "reason"), (
    ("reveal_record", "FINITE_RESPONSE_LAW_TARGET_REVEAL_REQUIRED"),
    ("analysis_record", "FINITE_RESPONSE_LAW_TARGET_ANALYSIS_REQUIRED"),
))
def test_finite_missing_grants_refuse_before_protected_parent_read(tmp_path, monkeypatch, missing, reason):
    from empirical_lawhood.adapters.composition import response_parent_reader

    parent = finite_retained_development_parent()
    roots = tuple(sorted(r for _, r in parent.root_identity_map))
    with TemporaryDirectory(prefix='finite-response-parent-joined-', dir="/dev/shm") as directory:
        target = Path(directory)
        (target / "artifacts").mkdir()
        profile = _profile(target)
        args, parent_path = publish_parent(target, profile=profile, repo=ROOT,
                                          route="finite-response-law-informative-composition", parent=parent, roots=roots)
        plan, prior = _finite_inputs(args["source_root"], roots)
        profile_path = tmp_path / "profile.json"
        profile_path.write_bytes(profile.canonical_bytes())
        args[missing] = None
        before = _snapshot(target)
        original = response_parent_reader.read_authenticated_parent
        entered = []

        def observed(**kwargs):
            entered.append(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(response_parent_reader, "read_authenticated_parent", observed)
        result = CliRunner().invoke(app, _finite_cli_args(profile_path, "informative-composition", args, plan=plan, prior=prior))
        assert result.exit_code == 3 and result.stdout == ""
        assert reason in result.stderr
        assert entered == []
        assert parent_path.is_file() and _snapshot(target) == before


@dataclass(frozen=True, slots=True)
class _OtherRetainedParent(FiniteResponseLawInformativeCompositionParent):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/tests/other-retained-parent'


@pytest.mark.parametrize(("fault", "reason", "protected_read_allowed"), (
    ("schema", "FINITE_RESPONSE_LAW_INFORMATIVE_COMPOSITION_TYPED_RETAINED_PARENT_REQUIRED", True),
    ("root-join", "FINITE_RESPONSE_LAW_INFORMATIVE_COMPOSITION_PARENT_PLAN_OR_ROOTS_MISMATCH", True),
    ("reveal-scope", "RESPONSE_PARENT_TARGET_GRANT_SCOPE_MISMATCH", False),
    ("analysis-scope", "RESPONSE_PARENT_TARGET_GRANT_SCOPE_MISMATCH", False),
    ("plan", "FINITE_RESPONSE_LAW_PLAN_MISMATCH", False),
))
def test_real_finite_retained_join_refusals(tmp_path, monkeypatch, fault, reason, protected_read_allowed):
    from empirical_lawhood.adapters.composition import response_parent_custody

    parent = finite_retained_development_parent()
    if fault == "schema":
        parent = _OtherRetainedParent(parent.science, parent.root_identity_map, parent.arrays_base64)
    roots = tuple(sorted(r for _, r in parent.root_identity_map))
    if fault == "root-join":
        roots = tuple(sorted(("synthetic.unjoined.prepared.r000", *roots[1:])))
    with TemporaryDirectory(prefix='finite-response-parent-joined-', dir="/dev/shm") as directory:
        target = Path(directory)
        (target / "artifacts").mkdir()
        profile = _profile(target)
        args, _ = publish_parent(target, profile=profile, repo=ROOT,
                                 route="finite-response-law-informative-composition", parent=parent, roots=roots)
        plan, prior = _finite_inputs(args["source_root"], roots)
        if fault.endswith("-scope"):
            field = "reveal_record" if fault == "reveal-scope" else "analysis_record"
            grant = args["authority_store"].resolve_grant(args[field])
            altered = replace(grant, grant_id="synthetic.wrong-scope", route="finite-response-law-calibration-method")
            args["authority_store"].persist(altered)
            args[field] = args[field].with_name(altered.grant_id + ".json")
        if fault == "plan":
            plan.write_bytes(plan.read_bytes() + b"\nchanged scientific plan\n")
        profile_path = tmp_path / "profile.json"
        profile_path.write_bytes(profile.canonical_bytes())
        original = response_parent_custody._read_member
        read_members = []

        def observed(root, path, **kwargs):
            read_members.append(path)
            return original(root, path, **kwargs)

        monkeypatch.setattr(response_parent_custody, "_read_member", observed)
        before = _snapshot(target)
        result = CliRunner().invoke(app, _finite_cli_args(profile_path, "informative-composition", args, plan=plan, prior=prior))
        assert result.exit_code == 3 and result.stdout == ""
        assert reason in result.stderr
        assert bool(read_members) is protected_read_allowed
        assert _snapshot(target) == before
