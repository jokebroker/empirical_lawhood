# SPDX-License-Identifier: MPL-2.0
"""Real finite public joins with fictional data in disposable stores only."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from typer.testing import CliRunner

from tests.response_custody_fixtures import publish_parent
from tests.finite_response_custody_fixtures import artifact, publish_runtime, qualified_parent
from tests.test_finite_response_assigned_evaluation_binding import _assigned_prospective_evaluation
from tests.test_response_joined_custody import ROOT, _finite_cli_args, _finite_inputs, _snapshot
from tests.test_doctor_profile import _profile
from empirical_lawhood.adapters.composition.finite_response_law.consumer_input import FiniteResponseLawAuthoringInput
from empirical_lawhood.adapters.methods.finite_response_law import law_payloads
from empirical_lawhood.cli.app import app
from empirical_lawhood.infrastructure.artifacts import (
    ExternalArtifactPlane,
    GuardedExternalRoot,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.operator_profile import resolve_external_root_contract


@pytest.fixture
def prospective_evaluation_join(tmp_path, request):
    with TemporaryDirectory(prefix="finite-public-", dir="/dev/shm") as directory:
        target = Path(directory)
        (target / "artifacts").mkdir()
        profile = _profile(target)
        plane = ExternalArtifactPlane(
            GuardedExternalRoot(
                resolve_external_root_contract(
                    profile, repo_root=ROOT, home_root=Path.home()
                )
            )
        )
        parent, native, native_receipt, native_artifact, native_receipt_artifact = (
            qualified_parent(plane)
        )
        if getattr(request, "param", None) == "negative":
            parent = replace(
                parent,
                calibration=replace(
                    parent.calibration,
                    joint_opportunities=tuple(
                        (name, False)
                        for name, _ in parent.calibration.joint_opportunities
                    ),
                ),
            )
        report_receipt = replace(
            native_receipt,
            receipt_id="synthetic.qualification.receipt",
            task_id="finite-response-law.calibration.qualify",
            output_materializations=(),
            output_logical_artifacts=(),
        )
        parent_artifact = artifact(
            "synthetic.report", parent.SCHEMA, parent.canonical_bytes()
        )
        report_receipt_artifact = artifact(
            report_receipt.receipt_id,
            report_receipt.SCHEMA,
            report_receipt.canonical_bytes(),
        )
        roots = tuple(r.stage_unit for r in native.config.projection.native_spec.roots)
        prior_units = tuple(
            r.physical_unit_id for r in native.config.projection.native_spec.roots
        )
        args, parent_path = publish_parent(
            target,
            profile=profile,
            repo=ROOT,
            route="finite-response-law-prospective-parent",
            parent=parent,
            roots=roots,
            members=(native, native_receipt, report_receipt),
        )
        assignment, source, control = _assigned_prospective_evaluation()
        control = replace(
            control,
            qualification=parent_artifact,
            qualification_receipt=report_receipt_artifact,
            expected_qualification_receipt=ObjectIdentity.from_record(
                report_receipt.receipt_id, report_receipt
            ),
            calibration_native=native_artifact,
            calibration_native_receipt=native_receipt_artifact,
            expected_calibration_native_receipt=ObjectIdentity.from_record(
                native_receipt.receipt_id, native_receipt
            ),
        )
        context = publish_runtime(plane, source)
        plan, prior = _finite_inputs(args["source_root"], prior_units)
        packet = FiniteResponseLawAuthoringInput(
            "prospective-evaluation",
            source,
            parent.fingerprint(),
            prior_units,
            ("synthetic.previous.stream",),
            control=control,
            runtime_context=context,
        )
        packet_path = args["source_root"] / "consumer.json"
        packet_path.write_bytes(packet.canonical_bytes())
        assignment_path = args["source_root"] / "assignment.json"
        assignment_path.write_bytes(assignment.canonical_bytes())
        profile_path = tmp_path / "profile.json"
        profile_path.write_bytes(profile.canonical_bytes())
        yield dict(
            target=target,
            plane=plane,
            profile=profile,
            args=args,
            parent=parent,
            parent_path=parent_path,
            packet=packet,
            packet_path=packet_path,
            assignment=assignment,
            assignment_path=assignment_path,
            profile_path=profile_path,
            plan=plan,
            prior=prior,
            native=native,
            native_receipt=native_receipt,
            report_receipt=report_receipt,
        )


def invoke_prospective_evaluation(data):
    return CliRunner().invoke(
        app,
        _finite_cli_args(
            data["profile_path"],
            data["packet"].stage,
            data["args"],
            plan=data["plan"],
            prior=data["prior"],
            assignment=data["assignment_path"],
            authoring_input=data["packet_path"],
        ),
    )


def test_assigned_qualification_accepts_exact_calibration_contract(prospective_evaluation_join):
    parent = prospective_evaluation_join["parent"]
    assert parent.eligible_for_prospective_evaluation
    assert len(parent.calibration.boundaries[0].root_ids) == 32
    assert all(b.q == 0 for b in parent.calibration.boundaries)
    lower = prospective_evaluation_join["plane"].root.resolve(
        parent.publications[0].authoritative_relative_locator, for_write=False
    )
    from empirical_lawhood.kernel.decoding import decode_canonical_bytes

    payload = decode_canonical_bytes(
        lower.read_bytes(), law_payloads.FiniteResponseLawLowerPayload, maximum_bytes=1024**2
    )
    for schema, version in (
        (payload.calibration.object_schema, "0.0.0"),
        ('empirical-lawhood/test/unknown-calibration', "1.0.0"),
    ):
        with pytest.raises(ValueError, match="calibration provenance"):
            replace(
                payload,
                calibration=replace(
                    payload.calibration, object_schema=schema, object_version=version
                ),
            )


def test_real_public_assigned_evaluation_join(prospective_evaluation_join):
    before = _snapshot(prospective_evaluation_join["target"])
    result = invoke_prospective_evaluation(prospective_evaluation_join)
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["binding_selected"] == 'FiniteResponseLawEvaluationSourceFactory'
    assert report["independent_units"] == 64 and report["native_tasks_planned"] == 1280
    assert len(report["selected_providers"]) == 5
    assert {owner["runner"] for owner in report["selected_providers"]} == {
        'FiniteResponseLawEvaluationSourceTask',
        'FiniteResponseLawProjectionTask',
        'FiniteResponseLawControlTask',
        'FiniteResponseLawEvaluationCompletionTask',
        'FiniteResponseLawEvaluationRevealTask',
    }
    assert (
        report["parent_custody_authenticated"]
        and report["target_reveal_authorized"]
        and report["target_analysis_authorized"]
    )
    assert report["evidence_role"] == "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
    assert report["native_tasks_executed"] == report["analysis_tasks_executed"] == 0
    assert not report["campaign_candidate_compiled"] and not report["campaign_issued"]
    assert not report["prospective_issue_eligible"]
    for owner in report["selected_providers"]:
        assert owner["runner"] and owner["provider_built"]
        assert {c["payload_schema"] for c in owner["output_semantic_contracts"]} == set(
            owner["output_schema_ids"]
        )
        assert all(c["capability_key"] for c in owner["output_semantic_contracts"])
    assert _snapshot(prospective_evaluation_join["target"]) == before


@pytest.mark.parametrize(
    ("fault", "reason", "protected_read"),
    (
        ("assignment-stage", "FINITE_RESPONSE_LAW_ASSIGNMENT_MISMATCH", False),
        ("assignment-namespace", "FINITE_RESPONSE_LAW_CONSUMER_ASSIGNMENT_MISMATCH", True),
        ("missing-reveal", "FINITE_RESPONSE_LAW_TARGET_REVEAL_REQUIRED", False),
        ("missing-analysis", "FINITE_RESPONSE_LAW_TARGET_ANALYSIS_REQUIRED", False),
        ("reveal-scope", "RESPONSE_PARENT_TARGET_GRANT_SCOPE_MISMATCH", False),
        ("analysis-scope", "RESPONSE_PARENT_TARGET_GRANT_SCOPE_MISMATCH", False),
        ("parent-bytes", "RESPONSE_PARENT_IDENTITY_BYTES_MISMATCH", True),
        ("incomplete-prior", "FINITE_RESPONSE_LAW_CURRENT_CENSUS_OMITS_ORIGINAL_PRIOR", True),
    ),
)
def test_assigned_join_first_refusal(
    prospective_evaluation_join, monkeypatch, fault, reason, protected_read
):
    from empirical_lawhood.adapters.composition import response_parent_custody
    from empirical_lawhood.adapters.composition.finite_response_law import consumer_ports

    if fault == "assignment-stage":
        from empirical_lawhood.adapters.composition.finite_response_law.assignment import proposed_scientific_seeds
        value = replace(
            prospective_evaluation_join["assignment"],
            stage="calibration",
            root_count=32,
            cohort_namespace="empirical-lawhood.finite-response-law.calibration.synthetic-other",
            scientific_seeds=proposed_scientific_seeds("calibration", 31337),
        )
        prospective_evaluation_join["assignment_path"].write_bytes(value.canonical_bytes())
    elif fault == "assignment-namespace":
        value = replace(
            prospective_evaluation_join["assignment"],
            cohort_namespace="empirical-lawhood.finite-response-law.prospective-evaluation.synthetic-other",
        )
        prospective_evaluation_join["assignment_path"].write_bytes(value.canonical_bytes())
    elif fault.startswith("missing-"):
        prospective_evaluation_join["args"][fault.removeprefix("missing-") + "_record"] = None
    elif fault.endswith("-scope"):
        field = fault.removesuffix("-scope") + "_record"
        store = prospective_evaluation_join["args"]["authority_store"]
        grant = store.resolve_grant(prospective_evaluation_join["args"][field])
        wrong = replace(grant, grant_id="synthetic.wrong-scope", route="finite-response-law-calibration-method")
        store.persist(wrong)
        prospective_evaluation_join["args"][field] = prospective_evaluation_join["args"][field].with_name(
            wrong.grant_id + ".json"
        )
    elif fault == "parent-bytes":
        prospective_evaluation_join["parent_path"].write_bytes(prospective_evaluation_join["parent_path"].read_bytes() + b" ")
    elif fault == "incomplete-prior":
        prior = json.loads(prospective_evaluation_join["prior"].read_bytes())
        prior["excluded_unit_ids"] = prior["excluded_unit_ids"][1:]
        prospective_evaluation_join["prior"].write_text(json.dumps(prior))
    reads = []
    ports = []
    original_read = response_parent_custody._read_member
    original_ports = consumer_ports.consumer_ports

    def observed_read(*args, **kwargs):
        reads.append(args)
        return original_read(*args, **kwargs)

    def observed_ports(**kwargs):
        ports.append(kwargs)
        return original_ports(**kwargs)

    monkeypatch.setattr(response_parent_custody, "_read_member", observed_read)
    monkeypatch.setattr(consumer_ports, "consumer_ports", observed_ports)
    before = _snapshot(prospective_evaluation_join["target"])
    refused = invoke_prospective_evaluation(prospective_evaluation_join)
    assert refused.exit_code == 3 and refused.stdout == "", refused.output
    assert reason in refused.stderr
    assert bool(reads) == protected_read
    assert not ports
    assert _snapshot(prospective_evaluation_join["target"]) == before


@pytest.mark.parametrize('prospective_evaluation_join', ["negative"], indirect=True)
def test_assigned_negative_eligibility_refuses_before_ports(prospective_evaluation_join, monkeypatch):
    from empirical_lawhood.adapters.composition.finite_response_law import consumer_ports

    assert not prospective_evaluation_join["parent"].eligible_for_prospective_evaluation
    entered = []
    original = consumer_ports.consumer_ports

    def observed(**kwargs):
        entered.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(consumer_ports, "consumer_ports", observed)
    before = _snapshot(prospective_evaluation_join["target"])
    refused = invoke_prospective_evaluation(prospective_evaluation_join)
    assert refused.exit_code == 3 and refused.stdout == ""
    assert "FINITE_RESPONSE_LAW_PROSPECTIVE_EVALUATION_QUALIFIED_PARENT_REQUIRED" in refused.stderr
    assert not entered and _snapshot(prospective_evaluation_join["target"]) == before


def continuation_join(data):
    from tests.finite_response_custody_fixtures import retained_continuation
    from empirical_lawhood.adapters.composition.finite_response_law.consumer_input import FiniteResponseLawAdditionalParent
    from empirical_lawhood.adapters.composition.finite_response_law.exposure import native_seed_ids

    source = data["packet"].source
    closeout, retention, members = retained_continuation(source)
    original = data["args"]
    additional = FiniteResponseLawAdditionalParent(
        *(
            str(original[field])
            for field in (
                "source_root",
                "parent_manifest",
                "custody",
                "reveal_record",
                "analysis_record",
            )
        )
    )
    args, parent_path = publish_parent(
        data["target"],
        profile=data["profile"],
        repo=ROOT,
        route="finite-response-law-prospective-parent",
        parent=closeout,
        roots=tuple(r.stage_unit for r in source.roots),
        source_name="continuation",
        artifact_members=members,
    )
    packet = replace(
        data["packet"],
        stage="prospective-continuation",
        primary_parent_sha256=closeout.fingerprint(),
        retention=retention,
        additional_parents=(additional,),
    )
    current_units = tuple(
        sorted(
            (
                *packet.original_prior_unit_ids,
                *(r.physical_unit_id for r in source.roots),
            )
        )
    )
    plan, prior = _finite_inputs(args["source_root"], current_units)
    census = json.loads(prior.read_bytes())
    census["excluded_seed_ids"] = sorted(
        (*packet.original_prior_seed_ids, *native_seed_ids(source))
    )
    prior.write_text(json.dumps(census))
    packet_path = args["source_root"] / "consumer.json"
    packet_path.write_bytes(packet.canonical_bytes())
    assignment_path = args["source_root"] / "assignment.json"
    assignment_path.write_bytes(data["assignment"].canonical_bytes())
    return dict(
        data,
        args=args,
        parent_path=parent_path,
        packet=packet,
        packet_path=packet_path,
        assignment_path=assignment_path,
        plan=plan,
        prior=prior,
    )


def test_real_public_continuation_reuses_original_roster(prospective_evaluation_join):
    data = continuation_join(prospective_evaluation_join)
    before = _snapshot(data["target"])
    result = invoke_prospective_evaluation(data)
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["binding_selected"] == 'FiniteResponseLawContinuationSourceFactory'
    assert report["independent_units"] == 64 and report["native_tasks_planned"] == 1216
    assert (
        report["streams_reused"]
        == len(json.loads(data["prior"].read_bytes())["excluded_seed_ids"]) - 1
    )
    assert len(report["parent_custody_sha256s"]) == 2
    assert {owner["runner"] for owner in report["selected_providers"]} == {
        "_ContinuationTask",
        "_PrefixImportTask",
        'FiniteResponseLawProjectionTask',
        'FiniteResponseLawControlTask',
        'FiniteResponseLawEvaluationCompletionTask',
        'FiniteResponseLawEvaluationRevealTask',
    }
    assert report["native_tasks_executed"] == report["analysis_tasks_executed"] == 0
    assert not report["campaign_issued"] and not report["prospective_issue_eligible"]
    assert _snapshot(data["target"]) == before


def test_continuation_incomplete_original_stream_census_refuses_before_parent(
    prospective_evaluation_join, monkeypatch
):
    from empirical_lawhood.adapters.composition import response_parent_custody

    data = continuation_join(prospective_evaluation_join)
    census = json.loads(data["prior"].read_bytes())
    census["excluded_seed_ids"] = census["excluded_seed_ids"][1:]
    data["prior"].write_text(json.dumps(census))
    entered = []
    original = response_parent_custody._read_member

    def observed(*args, **kwargs):
        entered.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(response_parent_custody, "_read_member", observed)
    before = _snapshot(data["target"])
    result = invoke_prospective_evaluation(data)
    assert result.exit_code == 3 and result.stdout == ""
    assert "FINITE_RESPONSE_LAW_RETAINED_ROSTER_REQUIRED" in result.stderr
    assert not entered and _snapshot(data["target"]) == before


def preparation_screen_join(data):
    from tests.finite_response_custody_fixtures import SyntheticSourceInventory
    from tests.test_finite_response_consumer_binding import stage_packet
    from empirical_lawhood.adapters.composition.finite_response_law.consumer_input import FiniteResponseLawAdditionalParent
    from empirical_lawhood.adapters.methods.finite_response_law.preparation_screen_results import FiniteResponseLawRetainedProspectiveCloseoutReference
    from empirical_lawhood.kernel.serialization import canonical_json_bytes

    baseline, _ = stage_packet("preparation-screening")
    rows = []
    prefix_members = []
    for row in baseline.source.retained_prefixes:
        declared = []
        for old in (*row.declaration.artifacts, row.declaration.task_receipt):
            raw = canonical_json_bytes(
                {
                    "schema": old.payload_schema,
                    "version": "1.0.0",
                    "value": {"synthetic_nonexecuting_fixture": old.artifact_id},
                }
            )
            identity = artifact(old.artifact_id, old.payload_schema, raw)
            prefix_members.append((identity, raw))
            declared.append(identity)
        declaration = replace(
            row.declaration, artifacts=tuple(declared[:-1]), task_receipt=declared[-1]
        )
        rows.append(
            replace(
                row,
                declaration=declaration,
                native_result=replace(
                    row.native_result, object_fingerprint=declared[0].sha256
                ),
            )
        )
    source = replace(baseline.source, retained_prefixes=tuple(rows))
    closeout = FiniteResponseLawRetainedProspectiveCloseoutReference(
        "cc1-finite-lawhood-v1.m4.formal-closeout",
        "0" * 64,
        1,
        "cc1-finite-lawhood-m4-formal-closeout-v1",
        True,
        "ADJUDICATED",
        True,
        "SUPPORTED",
        True,
        "0" * 40,
    )
    closeout_artifact = artifact(
        "synthetic.closeout", closeout.SCHEMA, closeout.canonical_bytes()
    )
    publication = data["parent"].publications[0]
    lower_raw = (
        data["plane"]
        .root.resolve(publication.authoritative_relative_locator, for_write=False)
        .read_bytes()
    )
    qualification = data["parent"].qualifications[0]
    qualification_artifact = artifact(
        "synthetic.lower.qualification",
        qualification.SCHEMA,
        qualification.canonical_bytes(),
    )
    old_adjudication = baseline.screen.prospective_adjudication
    adjudication_raw = canonical_json_bytes(
        {
            "schema": old_adjudication.payload_schema,
            "version": "1.0.0",
            "value": {"role": "SYNTHETIC_NONPROMOTABLE"},
        }
    )
    adjudication = artifact(
        old_adjudication.artifact_id, old_adjudication.payload_schema, adjudication_raw
    )
    screen = replace(
        baseline.screen,
        projection=replace(baseline.screen.projection, native_spec=source),
        lower_artifact=publication.artifact,
        lower_identity=ObjectIdentity(
            publication.artifact.artifact_id,
            publication.artifact.payload_schema,
            "1.0.0",
            publication.artifact.sha256,
        ),
        lower_qualification=qualification_artifact,
        prospective_adjudication=adjudication,
        prospective_closeout=closeout_artifact,
    )
    prepared_response_roots = tuple(
        sorted(r.root_id for r in source.roots if r.cohort == "prepared-response")
    )
    information_response_roots = tuple(
        sorted(
            r.root_id
            for r in source.roots
            if r.cohort == "information-response-prediction"
        )
    )
    prepared_response_prefix_artifact_ids = {
        a.artifact_id
        for r in source.roots
        if r.cohort == "prepared-response"
        for a in (
            *r.retained_prefix.declaration.artifacts,
            r.retained_prefix.declaration.task_receipt,
        )
    }
    information_response_prefix_artifact_ids = {
        a.artifact_id
        for r in source.roots
        if r.cohort == "information-response-prediction"
        for a in (
            *r.retained_prefix.declaration.artifacts,
            r.retained_prefix.declaration.task_receipt,
        )
    }
    # Each parent accounts for a genuine distinct subset of the 24 source units.
    extra, extra_path = publish_parent(
        data["target"],
        profile=data["profile"],
        repo=ROOT,
        route="finite-response-law-preparation-screening",
        parent=SyntheticSourceInventory(information_response_roots),
        roots=information_response_roots,
        source_name="preparation-screening-information-response",
        artifact_members=tuple(
            (a, raw) for a, raw in prefix_members if a.artifact_id in information_response_prefix_artifact_ids
        ),
    )
    args, parent_path = publish_parent(
        data["target"],
        profile=data["profile"],
        repo=ROOT,
        route="finite-response-law-preparation-screening",
        parent=closeout,
        roots=prepared_response_roots,
        source_name="preparation-screening-prepared-response",
        parent_artifact=closeout_artifact,
        artifact_members=tuple(
            (a, raw) for a, raw in prefix_members if a.artifact_id in prepared_response_prefix_artifact_ids
        )
        + (
            (publication.artifact, lower_raw),
            (qualification_artifact, qualification.canonical_bytes()),
            (adjudication, adjudication_raw),
        ),
    )
    additional = FiniteResponseLawAdditionalParent(
        *(
            str(extra[field])
            for field in (
                "source_root",
                "parent_manifest",
                "custody",
                "reveal_record",
                "analysis_record",
            )
        )
    )
    prior_units = tuple(
        sorted(("prior-unit", *(r.physical_unit_id for r in source.roots)))
    )
    packet = replace(
        baseline,
        source=source,
        primary_parent_sha256=closeout.fingerprint(),
        screen=screen,
        prospective_eligibility=ObjectIdentity.from_record(
            closeout_artifact.artifact_id, closeout
        ),
        additional_parents=(additional,),
    )
    plan, prior = _finite_inputs(args["source_root"], prior_units)
    prior_data = json.loads(prior.read_bytes())
    prior_data["excluded_seed_ids"] = ["prior-stream"]
    prior.write_text(json.dumps(prior_data))
    packet_path = args["source_root"] / "consumer.json"
    packet_path.write_bytes(packet.canonical_bytes())
    return dict(
        data,
        args=args,
        parent_path=parent_path,
        packet=packet,
        packet_path=packet_path,
        plan=plan,
        prior=prior,
        assignment_path=None,
        extra_path=extra_path,
    )


def test_current_preparation_policy_join_requires_eligibility_export(prospective_evaluation_join):
    data = preparation_screen_join(prospective_evaluation_join)
    before = _snapshot(data["target"])
    result = invoke_prospective_evaluation(data)
    assert result.exit_code == 3 and result.stdout == ""
    assert "FINITE_RESPONSE_LAW_CURRENT_PREPARATION_ELIGIBILITY_EXPORT_REQUIRED" in result.stderr
    assert _snapshot(data["target"]) == before


def test_preparation_policy_bad_additional_parent_refuses_before_ports(prospective_evaluation_join, monkeypatch):
    from empirical_lawhood.adapters.composition.finite_response_law import consumer_ports

    data = preparation_screen_join(prospective_evaluation_join)
    data["extra_path"].write_bytes(data["extra_path"].read_bytes() + b" ")
    entered = []
    original = consumer_ports.consumer_ports

    def observed(**kwargs):
        entered.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(consumer_ports, "consumer_ports", observed)
    before = _snapshot(data["target"])
    result = invoke_prospective_evaluation(data)
    assert result.exit_code == 3 and result.stdout == ""
    assert "RESPONSE_PARENT_IDENTITY_BYTES_MISMATCH" in result.stderr
    assert not entered and _snapshot(data["target"]) == before


@pytest.mark.parametrize(
    ("fault", "reason"),
    (
        ("unpublished", "FINITE_RESPONSE_LAW_PUBLISHED_RUNTIME_CONTEXT_REQUIRED"),
        ("source", "FINITE_RESPONSE_LAW_RUNTIME_SOURCE_MISMATCH"),
        ("subject", "FINITE_RESPONSE_LAW_RUNTIME_EXECUTION_AUTHORITY_MISMATCH"),
    ),
)
def test_real_runtime_binding_refuses_wrong_context(prospective_evaluation_join, fault, reason):
    packet = prospective_evaluation_join["packet"]
    if fault == "unpublished":
        identity = replace(
            packet.runtime_context, object_id="synthetic.unpublished-context"
        )
    else:
        identity = publish_runtime(
            prospective_evaluation_join["plane"],
            packet.source,
            context_id="synthetic.other-context",
            source_sha256="0" * 64 if fault == "source" else None,
            wrong_subject=fault == "subject",
        )
    packet = replace(packet, runtime_context=identity)
    prospective_evaluation_join["packet_path"].write_bytes(packet.canonical_bytes())
    before = _snapshot(prospective_evaluation_join["target"])
    result = invoke_prospective_evaluation(prospective_evaluation_join)
    assert result.exit_code == 3 and result.stdout == "", result.output
    assert reason in result.stderr
    assert _snapshot(prospective_evaluation_join["target"]) == before
