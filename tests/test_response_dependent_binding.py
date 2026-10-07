# SPDX-License-Identifier: MPL-2.0

"Public dependent refinement/fresh calibration path on explicitly synthetic, nonpromotable parents."

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest
from tests.response_dependent_fixtures import _selection_config
from tests.response_dependent_fixtures import calibration_records as calibration_records
from tests.response_dependent_fixtures import development_reports as development_reports
from tests.response_dependent_fixtures import qualification_fixture as qualification_fixture

from empirical_lawhood.adapters.composition.response_parent_custody import ResponseParentTargetGrant, ResponseSourceParentIdentity
from empirical_lawhood.adapters.composition.response_parent_input import ImportedResponseParentCustody
from empirical_lawhood.adapters.composition.response_parent_reader import AuthenticatedResponseParent
from empirical_lawhood.adapters.composition.prepared_response import dependent_input
from empirical_lawhood.adapters.composition.prepared_response.dependent_input import PreparedResponseDependentAuthoring, PreparedResponseDependentInput, check_dependent_input
from empirical_lawhood.adapters.composition.prepared_response.exposure import PreparedExposureInspection, prepared_seed_ids
from empirical_lawhood.adapters.composition.response_geometry_prospective.exposure import ResponseGeometryAssayExposureSource
from empirical_lawhood.adapters.methods.prepared_response.qualification import evaluate_prepared_response_source_qualification
from empirical_lawhood.adapters.simulators.prepared_response.extension_bundle import CALIBRATION_SOURCE_CAPABILITY, DEVELOPMENT_SOURCE_CAPABILITY
from empirical_lawhood.api.codecs import load_registered_authoring
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes

ROOT = Path(__file__).parents[1]


def _selection(route):
    return load_registered_authoring(
        ROOT / f"experiments/prepared-response/prepared-{route}-input.json",
        root_schemas={PreparedResponseDependentInput.SCHEMA: PreparedResponseDependentInput},
    )


def _exercise(
    tmp_path,
    monkeypatch,
    route,
    parent,
    native,
    parent_roots,
    parent_artifacts,
    costs=(),
):
    selection = _selection(route)
    grant = ResponseParentTargetGrant(
        "synthetic.custody",
        "CUSTODY",
        route,
        ResponseSourceParentIdentity(parent.SCHEMA, parent.VERSION, parent.fingerprint()),
        "f" * 64,
        "project",
        "runs/synthetic/recovery/resource-terminal.json",
        "runs/synthetic/outputs/parent.json",
        "runs/synthetic/outputs/stage.json",
        "synthetic-root",
        parent_roots,
        "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
    )
    custody = ImportedResponseParentCustody(
        parent.SCHEMA,
        grant.source_manifest_sha256,
        tuple(sorted(set(parent_artifacts))),
        parent_roots,
        ("c" * 64,),
        grant.evidence_role,
    )
    # This unit test substitutes the complete authenticated reader. The selector,
    # decoder, builder and provider execute below; test_response_joined_custody.py
    # separately joins them to real durable grants and custody replay.
    monkeypatch.setattr(
        dependent_input,
        "read_authenticated_parent",
        lambda **_: AuthenticatedResponseParent(grant, custody, parent.canonical_bytes()),
    )
    path = _authoring_input(tmp_path, route, native, parent_roots, costs)
    args = {
        "source_root": tmp_path,
        "parent_manifest": None,
        "custody": None,
        "reveal_record": None,
        "analysis_record": None,
        "repo_root": ROOT,
        "authoring_input": path,
    }
    return check_dependent_input(selection, **args), args, selection


def _authoring_input(tmp_path, route, native, parent_roots, costs=()):
    selection = _selection(route)
    capability = DEVELOPMENT_SOURCE_CAPABILITY if route == "dependent-refinement" else CALIBRATION_SOURCE_CAPABILITY
    plan_text = "Synthetic plan fixture; exposed and never issue eligible."
    source = replace(
        native,
        stage="development" if route == "dependent-refinement" else "calibration",
        seed_sha256=selection.source_seed_sha256,
        root_seed_census=(),
        implementation_plan_sha256=sha256(plan_text.encode()).hexdigest(),
        dependency_lock_sha256=sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        native_implementation=ObjectIdentity.from_record(
            capability.capability_key, capability
        ),
        selected_amplitude=Decimal(8),
    )
    sources = (ResponseGeometryAssayExposureSource("synthetic/prior.json", "e" * 64, 100, None, None),)
    exposure = PreparedExposureInspection(
        "synthetic.exposure",
        "2026-09-30T00:00:00Z",
        "d" * 64,
        source,
        ("synthetic",),
        sources,
        sha256(canonical_json_bytes(sources)).hexdigest(),
        parent_roots,
        ("synthetic.exposed.seed",),
        tuple(sorted(root.physical_unit_id for root in source.roots)),
        prepared_seed_ids(source),
        (),
        (),
        (),
    )
    authoring = PreparedResponseDependentAuthoring(
        plan_text, "Synthetic design fixture.", exposure, costs
    )
    path = tmp_path / "authoring-input.json"
    path.write_bytes(authoring.canonical_bytes())
    return path


def test_dependent_refinement_selected_builder_and_provider_and_negative_charter(
    tmp_path, monkeypatch, qualification_fixture, development_reports
):
    config, observations = qualification_fixture
    parent = evaluate_prepared_response_source_qualification(config, observations)
    roots = tuple(sorted({item.root.root_id for item in observations}))
    artifacts = tuple(item.object_fingerprint for item in parent.observations)
    costs = _selection_config(development_reports[0]).candidate_costs
    report, args, selection = _exercise(
        tmp_path,
        monkeypatch,
        "dependent-refinement",
        parent,
        config.projection.native_spec,
        roots,
        artifacts,
        costs,
    )
    assert report["provider_built"] and report["independent_units"] == 128
    assert report["factory"] == 'PreparedResponseDevelopmentSourceFactory'
    assert report["native_tasks_executed"] == 0 and not report["campaign_issued"]
    assert report["prospective_issue_eligible"] is False
    trusted = dependent_input.read_authenticated_parent()
    negative = replace(
        parent,
        screens=tuple(
            replace(screen, two_direction_contact_roots=0, distinct_command_roots=0)
            for screen in parent.screens
        ),
    )
    monkeypatch.setattr(
        dependent_input,
        "read_authenticated_parent",
        lambda **_: replace(trusted, raw=negative.canonical_bytes()),
    )
    with pytest.raises(ValueError, match="NO_COMMON_QUALIFIED_CHARTER"):
        check_dependent_input(selection, **args)


def test_fresh_response_calibration_selected_builder_and_provider(tmp_path, monkeypatch, calibration_records):
    records, _, _ = calibration_records
    source, policy, _, _ = records
    parent = policy.development_library
    assert parent.disposition == "NOMINATED_FOR_FRESH_CALIBRATION"
    roots = tuple(
        sorted(root.root_id for root in parent.config.fit.projection.native_spec.roots)
    )
    artifacts = tuple(item.object_fingerprint for item in parent.fits)
    report, _, _ = _exercise(
        tmp_path, monkeypatch, "fresh-response-calibration", parent, source, roots, artifacts
    )
    assert report["provider_built"] and report["independent_units"] == 192
    assert report["factory"] == 'PreparedResponseCalibrationSourceFactory'
    assert report["native_tasks_executed"] == 0 and not report["campaign_issued"]
