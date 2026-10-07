"""Complete Q8/E32 software custody, synthetic signers and engineered operands.

No native Q/E campaign is performed. Only the numerical source-read seam supplies
engineered arrays; current issue/package/receipts/recovery/grants and all owned
request, validity, constructor, original readout and diagnostic arithmetic run.
"""

import base64
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256

import numpy as np
import pytest

from empirical_lawhood.api import preparation_diagnostics as api
from empirical_lawhood.api.preparation_inputs import bind_constructed_preparation_q
from empirical_lawhood.api.preparation_result_access import preparation_result_access
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.diagnostic_records import PreparationDiagnosticInput
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.records import ConstructedPreparationSelection, ConstructedPreparationLowerSeal, numbers
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.measurement import measure_root
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.readout import report, constructor_crossing
from empirical_lawhood.adapters.methods.preparation_applicability import measurement
from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.records import ConstructedPreparationNativePhase, ConstructedPreparationStream, ConstructedPreparationPrefix, ConstructedPreparationParents, ConstructedPreparationPanel, nominal_digest
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.recovery import ExternalRunRecoveryStore
from empirical_lawhood.infrastructure.study_issue import ExternalStudyOperationAuthorityStore
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import OperationalStatus, ScientificStatus, AdmissionStatus
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord, AdjudicationEvaluability
from empirical_lawhood.runtime.execution import OperationalAttempt, TaskAttemptDisposition
from empirical_lawhood.runtime.recovery import build_run_recovery_index
from tests.finite_response_rerun_fixtures import publish_current_issue, publish_current_step_output
from tests.test_preparation_applicability_public_application import _setup, _selection, _author_load_prove, actual_short_conformance as _actual_short_conformance


@pytest.fixture(scope="module")
def diagnostic_conformance(tmp_path_factory):
    return _actual_short_conformance.__wrapped__(tmp_path_factory)


def _phase(root, source, *, phase, view, incoming=None, schedule=None, future=None, word=None):
    """Explicit unexecuted compact software source record, never native evidence."""
    purpose = f"future-{future + 1}" if phase == "future" else phase
    streams = tuple(ConstructedPreparationStream(root.seed_for(name),
        None if phase == "future" else nominal_digest(name), Decimal(1) if phase == "future" else Decimal(".002"))
        for name in (purpose, purpose + "-bridge"))
    clocks = {"prefix": (0, 4096), "parent": (4096, 4496), "future": (4496, 4688)}
    start, end = clocks[phase]
    return ConstructedPreparationNativePhase(root.root_id, phase, schedule, future, word, view,
        source.fingerprint(), incoming, start, end, 0, 0, (Decimal(0), Decimal(0)), Decimal(0), Decimal(0), Decimal(0),
        "NUMERICAL_FAILURE", "unexecuted engineered source-read software fixture", sha256(f"{root.root_id}.{purpose}".encode()).hexdigest(),
        streams, "", (), "", "", "", (), None, (), f"software-history.{root.root_id}.view-{view}", "", allocation=root)


def _engineered_arrays(lower):
    z = np.tile(np.asarray(lower.center, dtype=float)[None, :, None], (3, 1, 2))
    for schedule, scaled in enumerate((7.0, 5.0, 5.5)):
        z[schedule, 17, :] += scaled * float(lower.scale[17])
    # Exact paired views; deliberately broad lower accuracy/support gates may fail.
    # Q constructor qualification adds no such extra gate.
    y = np.zeros((3, 4, 8, 2, 2))
    work = np.zeros((3, 2))
    return z, y, work


def _root_records(stage, source, lower, root):
    prefix = ConstructedPreparationPrefix(root.root_id, tuple(_phase(root, source, phase="prefix", view=v) for v in (1, 2)),
        base64.b64encode(np.zeros((2, 3, 4, 4), dtype=np.complex128).tobytes()).decode(),
        (tuple(lower.center), tuple(lower.center)))
    selected = ConstructedPreparationSelection(root.root_id, prefix.fingerprint(), None, 1, numbers((0, 1, 0)))
    parents = ConstructedPreparationParents(root.root_id, prefix.fingerprint(), selected.fingerprint(),
        tuple(_phase(root, source, phase="parent", view=v, incoming=prefix.phases[v - 1].fingerprint(), schedule=s)
            for s in range(3) for v in (1, 2)))
    z, y, work = _engineered_arrays(lower)
    _, _, _, mean, width, support = measurement.validity(lower, z, y, work)
    direction, requirement = measurement.requests(root.seed_for("requests"))
    sealed = ConstructedPreparationLowerSeal(root.root_id, parents.fingerprint(), lower.fingerprint(), True,
        numbers(mean), numbers(width), tuple(map(bool, support)),
        tuple(map(int, measurement.lower_choices(mean, width, support, direction, requirement).ravel())),
        tuple(map(int, direction.ravel())), numbers(requirement))
    futures = tuple(_phase(root, source, phase="future", view=v,
        incoming=next(p.fingerprint() for p in parents.phases if (p.schedule_index, p.refinement) == (s, v)),
        schedule=s, future=f, word=w) for s in range(3) for v in (1, 2) for f in (0, 1) for w in range(9))
    panel = ConstructedPreparationPanel(root.root_id, prefix.fingerprint(), selected.fingerprint(), (*parents.phases, *futures), sealed.fingerprint())
    measured = measure_root(prefix, panel, lower, stage.phase, root.seed_for("requests"), sealed)
    assert measured.complete
    return prefix, selected, parents, sealed, panel, measured


def _publish_phase(selection, handoff, plane, issue):
    lower = next(value.record for value in selection.upstream_records if value.key == "lower")
    cells = tuple(_root_records(selection.stage, selection.source, lower, root) for root in selection.stage.allocation.roots)
    summary = report(selection.stage, tuple(records[-1] for records in cells),
        tuple(constructor_crossing(records[-1], lower) for records in cells))
    task_records = {f"ap.{task}.{root.root_id}": record for root, records in zip(selection.stage.allocation.roots, cells, strict=True)
        for task, record in zip(api.TASKS, records, strict=True)}
    task_records["ap.report"] = summary
    receipts, outputs = {}, {}
    tasks = {task.task_id: task for task in issue.execution_plan.tasks}
    for ordinal, task_id in enumerate(issue.execution_plan.topological_task_ids()):
        task = tasks[task_id]
        dependency_materializations = tuple(sorted(materialization.materialization_id
            for dependency in task.dependency_task_ids for materialization in receipts[dependency].output_materializations))
        if task.task_id == "ap.adjudication":
            package = issue.package
            status = ScientificStatus.SUPPORTED if summary.disposition in ("QUALIFIED", "SUPPORTED") else ScientificStatus.NOT_SUPPORTED
            record = ScientificAdjudicationRecord(f"{issue.publication.run_id}.adjudication", issue.publication.run_id,
                task.task_id, ObjectIdentity.from_record(issue.execution_plan.execution_plan_id, issue.execution_plan),
                dependency_materializations, tuple(sorted(o.logical_artifact_id for o in task.outputs)),
                tuple(sorted(receipts[d].receipt_id for d in task.dependency_task_ids)),
                package.system.world.world_id, package.system.world.kind,
                ObjectIdentity.from_record(package.experiment.relation.relation_id, package.experiment.relation),
                package.experiment.independent_unit_id,
                tuple(ObjectIdentity.from_record(c.cutoff_id, c) for c in package.experiment.information_cutoffs),
                task.outputs[0].visibility_ceiling, task.outputs[0].outcome_access,
                AdjudicationEvaluability.EVALUABLE, status, AdmissionStatus.NOT_EVALUATED,
                (f"BOUNDARY_{selection.stage.phase}_{summary.disposition}",))
        else:
            record = task_records[task.task_id]
        receipt, manifests, _ = publish_current_step_output(plane, issue, label=f"software-{selection.stage.phase.lower()}-{ordinal}",
            record=record, task_id=task.task_id, input_materialization_ids=dependency_materializations)
        receipts[task.task_id] = receipt
        outputs[task.task_id] = manifests
    recovery = ExternalRunRecoveryStore(plane, minimum_free_bytes=1)
    index = build_run_recovery_index(issue.execution_plan, run_id=issue.publication.run_id, wave_id="software-wave-001",
        authority_identities=tuple(sorted((issue.publication.issued_study, issue.publication.execution_authority,
            issue.publication.reveal_authority), key=lambda i: i.object_id)))
    recovery.freeze(index)
    for receipt in receipts.values():
        recovery.record_receipt(index, receipt)
    attempts = tuple(OperationalAttempt(receipt.attempt_id, receipt.run_id, receipt.task_id, 1,
        TaskAttemptDisposition.SUCCEEDED, None, None, None) for receipt in receipts.values())
    recovery.record_terminal(index, status=OperationalStatus.SUCCEEDED, attempts=attempts, receipts=tuple(receipts.values()))
    access = preparation_result_access(run_id=issue.publication.run_id, issued_study_id=issue.publication.issued_study.object_id,
        execution_authority_id=issue.publication.execution_authority.object_id, reveal_authority_id=issue.publication.reveal_authority.object_id, writer=plane)
    phase = api.prepare_preparation_diagnostic_phase(selection=selection, result_access=access,
        recovery_index_relative_path=index.index_relative_path, artifact_writer=plane)
    return phase, summary, receipts, outputs


def test_complete_current_q8_e32_issue_receipts_diagnostics_recovery_and_prebyte_denials(tmp_path, monkeypatch, diagnostic_conformance):
    plane, publications, _, boundary = _setup(tmp_path, monkeypatch, diagnostic_conformance)
    lower = publications["lower"].record
    def engineered_read(prefix, panel):
        assert panel.root_id == prefix.root_id and panel.prefix_sha256 == prefix.fingerprint()
        assert len(panel.phases) == 114
        return _engineered_arrays(lower)
    monkeypatch.setattr(measurement, "reduce_panel", engineered_read)
    monkeypatch.setattr("empirical_lawhood.adapters.methods.constructed_preparation_applicability.measurement.reduce_panel", engineered_read)
    monkeypatch.setattr(api, "reduce_panel", engineered_read)
    q = _selection("Q", True, publications)
    q_handoff = _author_load_prove(boundary, q, plane, True)
    q_issue = publish_current_issue(plane, q_handoff, label="diagnostics-q8")
    q_phase, q_report, q_receipts, _ = _publish_phase(q, q_handoff, plane, q_issue)
    assert q_report.constructor_qualified and len(q_report.root_ids) == 8
    qualified = bind_constructed_preparation_q(run_id=q_issue.publication.run_id,
        receipt_id=q_receipts["ap.report"].receipt_id, artifact_writer=plane, result_access=q_phase.result_access)
    e = _selection("E", True, publications, qualified)
    e_handoff = _author_load_prove(boundary, e, plane, True)
    e_issue = publish_current_issue(plane, e_handoff, label="diagnostics-e32")
    e_phase, e_report, _, _ = _publish_phase(e, e_handoff, plane, e_issue)
    assert e_report.complete and len(e_report.root_ids) == 32
    config = PreparationDiagnosticInput("synthetic.current.diagnostics", q_phase, e_phase,
        api.current_preparation_diagnostic_source_sha256(), 884001, 884002)
    # Preserve the actual software issue/grant/census inputs for bounded fresh-
    # process denial checks without rebuilding the full Q8/E32 publication.
    (tmp_path / "diagnostic-input.json").write_bytes(config.canonical_bytes())
    result, inputs = api.analyze_preparation_diagnostics(config, artifact_writer=plane)
    assert len(inputs) == 244 and result.native_calls == 0
    published = api.publish_preparation_diagnostics(config, artifact_writer=plane, relative_root="software/diagnostics")
    assert published.logical.outcome_access is OutcomeAccess.EVALUATION_REVEALED
    monkeypatch.setattr(api, "analyze_preparation_diagnostics", lambda *_a, **_k: pytest.fail("read reran diagnostic arithmetic"))
    fresh = ExternalArtifactPlane(plane.root)
    reread = api.read_preparation_diagnostics(config, artifact_writer=fresh, relative_root="software/diagnostics")
    assert reread["report_sha256"] == result.fingerprint()
    assert reread["phases"]["Q"]["n_independent_roots"] == 8 and reread["phases"]["E"]["n_independent_roots"] == 32
    assert reread["phases"]["E"]["probability_bounds"]["full_valid_roots"] == e_report.full_valid_counts[1]
    with pytest.raises(ValueError, match="source"):
        api.read_preparation_diagnostics(replace(config, analysis_source_sha256="0" * 64), artifact_writer=fresh, relative_root="software/diagnostics")
    with pytest.raises(ValueError, match="census"):
        replace(e_phase, roots=e_phase.roots[:-1])
    wrong_access = replace(e_phase.result_access, reveal_authority=replace(e_phase.result_access.reveal_authority, object_fingerprint="a" * 64))
    denied = replace(config, evaluation=replace(e_phase, result_access=wrong_access))
    protected = []
    original_verify = fresh.verify_manifest
    def before_bytes(manifest):
        if manifest.logical.outcome_access is OutcomeAccess.EVALUATION_REVEALED:
            protected.append(manifest.materialization.relative_path)
        return original_verify(manifest)
    monkeypatch.setattr(fresh, "verify_manifest", before_bytes)
    with pytest.raises(PermissionError):
        api.prepare_preparation_diagnostic_phase(selection=e, result_access=wrong_access,
            recovery_index_relative_path=e_phase.recovery_index_relative_path, artifact_writer=fresh)
    assert not protected
    with pytest.raises(PermissionError):
        api.read_preparation_diagnostics(denied, artifact_writer=fresh, relative_root="software/diagnostics")
    assert not protected
    assert ExternalStudyOperationAuthorityStore(fresh).load(e_phase.result_access.reveal_authority.object_id) is not None
