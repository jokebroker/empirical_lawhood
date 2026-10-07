"""Full-size noncontact scan proof and bounded actual development-native cells."""

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from empirical_lawhood.api import matrix_geometry as api
from empirical_lawhood.api.matrix_native_custody import MatrixNativeCustody
from empirical_lawhood.adapters.methods.matrix_response_study import anisotropic_feasibility as science
from empirical_lawhood.adapters.methods.matrix_response_study import numerical_qualification as qualification
from empirical_lawhood.adapters.simulators.six_matrix_response.extension_bundle import SIX_MATRIX_RESPONSE_CAPABILITY
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import SixMatrixResponseEpisodeTerminal
from empirical_lawhood.adapters.methods.matrix_response_study.geometry_inputs import allocate_matrix_geometry, MatrixGeometrySeedCell
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import SixMatrixResponseConstitution, derive_rng_stream
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.runtime.artifacts import ExternalRootContract
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plane(tmp_path):
    store = tmp_path / "synthetic-matrix-store"
    store.mkdir()
    return ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract(
        "synthetic.matrix-store", "software conformance storage", str(store), "/",
        OperatorStorageProfile.SCHEMA, 1, None, None, ())))


@pytest.fixture
def config():
    return replace(api.matrix_geometry_example_input(environment_lock_sha256=sha256((ROOT / "uv.lock").read_bytes()).hexdigest()), minimum_free_bytes=1)


def _factor(name, geometric=False):
    return science.MatrixResponseAnisotropicFeasibilityFactorDiagnostic(name, Decimal(4), Decimal("0.66"),
        Decimal("0.1") if geometric else Decimal("0.5"), Decimal("0.1") if geometric else Decimal("0.5"),
        Decimal(1) if geometric else Decimal(0), 16, 1, geometric, () if geometric else ("synthetic-nongeometric",))


def _row(task, *, constitution="00"):
    return science.MatrixResponseAnisotropicFeasibilityRolloutSummary(science._rollout_id(task), task.stage,
        task.member.member_id, task.numerical_view.view_id, task.q, task.alpha_x_index, task.alpha_y_index,
        task.alpha_tilde_x, task.alpha_tilde_y, task.history_id, task.seed_index,
        task.schedule.total_steps * task.step_multiplier, task.schedule.total_steps * task.step_multiplier,
        task.scientific_seed_sha256, "1" * 64, _factor("X", constitution[0] == "1"), _factor("Y", constitution[1] == "1"),
        SixMatrixResponseConstitution(constitution), True, ())


def test_full_scan_proof_has_exact_roots_nested_views_resources_and_no_native_contact(config, monkeypatch):
    monkeypatch.setattr(science, "_run_rollout", lambda *_: pytest.fail("noncontact proof entered native science"))
    proof = api.prove_matrix_geometry_scan(config)
    assert proof["feasibility_roots"] == 9126
    assert proof["potential_task_count"] == 17118
    assert proof["actual_selected_confirmation_roots"] == 1296
    assert proof["secondary_nested_views"] == 36
    assert proof["maximum_integration_steps"] == 12146688
    assert proof["exposed_example"] and not proof["scientific_execution_performed"]
    assert decode_canonical_bytes(config.canonical_bytes(), type(config), maximum_bytes=api.MATRIX_GEOMETRY_INPUT_MAXIMUM_BYTES) == config


def test_allocations_refuse_missing_duplicate_history_coordinates_and_effective_reuse(config):
    allocation = config.allocation
    with pytest.raises(ValueError, match="census"):
        replace(allocation, cells=allocation.cells[:-1])
    with pytest.raises(ValueError, match="census"):
        replace(allocation, cells=(allocation.cells[0], *allocation.cells[:-1]))
    with pytest.raises(ValueError, match="stream"):
        replace(allocation, cells=(allocation.cells[0], replace(allocation.cells[1], full_seed_sha256=allocation.cells[0].full_seed_sha256[:32] + "f" * 32), *allocation.cells[2:]))
    with pytest.raises(ValueError, match="exposed"):
        replace(allocation, prior_exposed_seed_sha256s=(allocation.cells[0].full_seed_sha256,))
    bad = MatrixGeometrySeedCell((0, 0, 0, 0, 1, 0, 0), "0" * 64)
    with pytest.raises(ValueError, match="census"):
        replace(allocation, cells=(bad, *allocation.cells[1:]))


def test_label_changes_preserve_actual_draws_and_scientific_seed_changes_change_them(config):
    renamed = replace(config.allocation, allocation_id="synthetic.renamed")
    other = allocate_matrix_geometry(allocation_id="synthetic.other", master_seed=473847)
    def draws(seed, label):
        rng, _ = derive_rng_stream(seed_root_id=label, purpose_id=label, stream_index=0,
            derivation_rule_id="six-matrix-response.rng.explicit-original-full-seed-pcg64dxsm", scientific_seed_sha256=seed)
        return rng.standard_normal(12)
    assert np.array_equal(draws(config.allocation.cells[0].full_seed_sha256, "original"), draws(renamed.cells[0].full_seed_sha256, "renamed"))
    assert not np.array_equal(draws(config.allocation.cells[0].full_seed_sha256, "original"), draws(other.cells[0].full_seed_sha256, "original"))


def test_actual_native_cell_retains_negative_geometry_and_recovers_without_native_repetition(config, plane, monkeypatch):
    kw = dict(config=config, coordinate=(0, 0, 0, 0, 0, 0, 0), artifact_writer=plane, project_root=ROOT, implementation_commit="0" * 40)
    row = api.run_matrix_geometry_conformance(**kw)
    assert row.valid and row.completed_steps == 1024 and row.constitution.value == "00"
    monkeypatch.setattr(science, "_run_rollout", lambda *_: pytest.fail("completed numerical cell repeated"))
    fresh = ExternalArtifactPlane(plane.root)
    assert api.run_matrix_geometry_conformance(**{**kw, "artifact_writer": fresh}) == row
    task = next(t for t in api._all_tasks(config) if science._rollout_id(t) == row.rollout_id)
    with pytest.raises(ValueError, match="history/root/view/seed"):
        api._validate_rollout(task, replace(row, history_id="synthetic.other-history"))


def test_current_source_lock_and_prerequisite_refuse_before_native_contact(config, plane, monkeypatch):
    monkeypatch.setattr(science, "_run_rollout", lambda *_: pytest.fail("invalid source entered native science"))
    with pytest.raises(ValueError, match="source owners"):
        api.prove_matrix_geometry_scan(replace(config, scientific_code_sha256="0" * 64))
    with pytest.raises(ValueError, match="lock"):
        api.run_matrix_geometry_conformance(config=replace(config, environment_lock_sha256="0" * 64), coordinate=(0, 0, 0, 0, 0, 0, 0), artifact_writer=plane, project_root=ROOT, implementation_commit="0" * 40)
    with pytest.raises(ValueError, match="numerical qualification"):
        api.run_matrix_geometry_scan(config=config, artifact_writer=plane, project_root=ROOT, implementation_commit="0" * 40)
    with pytest.raises(TypeError, match="guarded"):
        api.run_matrix_geometry_scan(config=config, artifact_writer=None, project_root=ROOT, implementation_commit="0" * 40)


def test_interrupted_contact_and_unreceipted_publication_cannot_grant_native_retry(config, plane):
    custody = MatrixNativeCustody(writer=plane, config=config, run_id=config.run_id,
        implementation_commit="0" * 40, minimum_free_bytes=1, maximum_output_bytes=config.maximum_output_bytes)
    custody.begin_native_effect("synthetic-interrupted")
    with pytest.raises(ValueError, match="uncertain effects"):
        custody.load("synthetic-interrupted", science.MatrixResponseAnisotropicFeasibilityRolloutSummary)
    with pytest.raises(ValueError, match="no retry"):
        custody.begin_native_effect("synthetic-interrupted")


def test_no_candidate_is_complete_negative_terminal_and_never_runs_confirmation(config):
    calls = []
    def execute(tasks, *, workers, progress):
        calls.extend(tasks)
        assert all(t.stage is science.MatrixResponseAnisotropicFeasibilityStage.FEASIBILITY for t in tasks)
        return tuple(reversed(tuple(_row(t) for t in tasks)))
    report = science.run_matrix_response_study_anisotropic_feasibility_feasibility(source=config.source,
        implementation_commit="0" * 40, numerical_qualification_report_sha256="0" * 64,
        allocation=config.allocation, workers=1, task_executor=execute)
    assert len(calls) == len(report.rollouts) == 9126
    assert report.selected_member_id is None and not report.view_comparisons
    assert report.disposition is science.MatrixResponseAnisotropicFeasibilityDisposition.NO_CONSTITUTIVE_SUBSTRATE
    assert report.evidence_ceiling is EvidenceCeiling.NON_PROMOTABLE
    assert report.outcome_access is OutcomeAccess.DEVELOPMENT_VISIBLE
    assert report.visibility_ceiling is VisibilityCeiling.DEVELOPMENT_ONLY
    assert report.source_config == ObjectIdentity.from_record(config.source.config_id, config.source)


def test_recurrence_axial_erosion_and_noncompensating_numeric_member_tie_break(config):
    template = api._all_tasks(config)[0]
    rows = tuple(_row(replace(template, alpha_x_index=x, alpha_y_index=y, history_id=f"synthetic.h{history}", seed_index=seed), constitution="10")
        for x, y in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)) for history in range(2) for seed in range(2))
    support = science.aggregate_cell_support(rows, stage=science.MatrixResponseAnisotropicFeasibilityStage.FEASIBILITY, seed_minimum=2, history_minimum=2)
    assert next(r for r in support if (r.alpha_x_index, r.alpha_y_index) == (1, 1)).interior
    assert not next(r for r in support if (r.alpha_x_index, r.alpha_y_index) == (0, 1)).interior
    def score(name, eligible, width):
        return science.MatrixResponseAnisotropicFeasibilityMemberScore(f"score.{name}", name, eligible, width, 10, 8, Decimal(1), Decimal("0.9"), 0, False)
    scores = (score("z", False, 99), score("a", True, 2), score("b", True, 2))
    assert tuple(r.member_id for r in science.select_feasibility_member(scores, scientific_member_order=("b", "a", "z")) if r.selected) == ("b",)
    assert science.select_feasibility_member(tuple(reversed(scores)), scientific_member_order=("b", "a", "z")) == science.select_feasibility_member(scores, scientific_member_order=("b", "a", "z"))


def _fictional_qualification(config):
    """Complete engineered prerequisite operands; no numerical qualification claim."""
    checks = tuple(qualification.MatrixResponseNumericalQualificationCheck(
        f"matrix-response-numerical-qualification.{name}", Decimal(0), Decimal(1), "LE", True, ())
        for name in sorted(api._NUMERICAL_CHECK_NAMES))
    phase = qualification.MatrixResponseNumericalQualificationPhaseLabel.GEOMETRIC_PRODUCT
    points = tuple(qualification.MatrixResponseNumericalQualificationCanonicalPoint(
        f"synthetic.point-{i:03d}", ObjectIdentity.from_record(model.model_id, model),
        SixMatrixResponseEpisodeTerminal.COMPLETED, phase, (phase,), Decimal(1), Decimal(1), Decimal(0), Decimal(0), True)
        for i, model in enumerate(config.source.canonical_reference_models))
    benchmarks = tuple(qualification.MatrixResponseNumericalQualificationBenchmark(
        f"synthetic.benchmark-q{q}", q, 256, Decimal(1), Decimal(256), 1024, Decimal("0.1")) for q in (2, 3, 4))
    return qualification.MatrixResponseNumericalQualificationQualificationReport("synthetic.numerical-products", "0" * 40,
        ObjectIdentity.from_record(config.source.config_id, config.source),
        ObjectIdentity.from_record(SIX_MATRIX_RESPONSE_CAPABILITY.capability_key, SIX_MATRIX_RESPONSE_CAPABILITY),
        checks, points, benchmarks, qualification.MatrixResponseNumericalQualificationDisposition.NUMERICALLY_QUALIFIED,
        (), EvidenceCeiling.NON_PROMOTABLE, OutcomeAccess.DEVELOPMENT_VISIBLE, VisibilityCeiling.DEVELOPMENT_ONLY, False)


def test_public_complete_negative_terminal_actual_custody_fresh_read_and_source_root_refusals(config, plane, monkeypatch):
    """Full software publication/recovery with engineered rows; no scan is acquired."""
    fake_qualification = _fictional_qualification(config)
    monkeypatch.setattr(qualification, "run_matrix_response_study_numerical_qualification_qualification", lambda **_: fake_qualification)
    api.run_matrix_numerical_qualification(config=config, artifact_writer=plane, project_root=ROOT, implementation_commit="0" * 40)
    bound = api.bind_matrix_geometry_qualification(config=config, qualification_config=config,
        artifact_writer=plane, implementation_commit="0" * 40)
    calls = []
    def engineered_cell(task):
        calls.append(task)
        assert task.stage is science.MatrixResponseAnisotropicFeasibilityStage.FEASIBILITY
        return _row(task)
    with monkeypatch.context() as acquisition:
        acquisition.setattr(science, "_run_rollout", engineered_cell)
        result = api.run_matrix_geometry_scan(config=bound, artifact_writer=plane, project_root=ROOT, implementation_commit="0" * 40)
    assert len(calls) == len(result["report"].rollouts) == 9126
    assert result["report"].selected_member_id is None
    fresh = ExternalArtifactPlane(plane.root)
    with monkeypatch.context() as retained_read:
        retained_read.setattr(science, "_run_rollout", lambda *_: pytest.fail("read repeated engineered acquisition"))
        reread = api.read_matrix_geometry_result(config=bound, artifact_writer=fresh, implementation_commit="0" * 40)
        assert reread["report"] == result["report"]
        with pytest.raises(ValueError, match="source owners"):
            api.read_matrix_geometry_result(config=replace(bound, scientific_code_sha256="0" * 64), artifact_writer=fresh, implementation_commit="0" * 40)
        custody = MatrixNativeCustody(writer=fresh, config=bound, run_id=bound.run_id, implementation_commit="0" * 40,
            minimum_free_bytes=1, maximum_output_bytes=bound.maximum_output_bytes, read_only=True)
        partial = replace(result["report"], rollouts=result["report"].rollouts[:-1])
        original_load = MatrixNativeCustody.load
        def missing_root(self, task_id, record_type, **kwargs):
            return partial if self.run_id == bound.run_id and task_id == "terminal" else original_load(self, task_id, record_type, **kwargs)
        with monkeypatch.context() as missing:
            missing.setattr(MatrixNativeCustody, "load", missing_root)
            with pytest.raises(ValueError, match="requested root"):
                api.read_matrix_geometry_result(config=bound, artifact_writer=fresh, implementation_commit="0" * 40)
        assert custody.input_manifest.logical.content_sha256 == bound.fingerprint()
    _selected_view_sensitive_public_case(bound, fresh, monkeypatch)


def test_selected_view_sensitive_continuation_acquires_one_parent_and_recovers(config, plane, monkeypatch):
    """Exercise the separate parent/bridge continuation without a durable null scan."""
    from empirical_lawhood.api import selected_events
    from tests.test_selected_event_integration import _reference_report
    from types import SimpleNamespace

    original_rollout = science._run_rollout
    calls = []
    def observed_rollout(task):
        calls.append(task)
        return original_rollout(task)
    monkeypatch.setattr(science, "_run_rollout", observed_rollout)
    request = replace(selected_events.selected_parent_example_input(environment_lock_sha256=config.environment_lock_sha256), minimum_free_bytes=1)
    cached_reference = None
    def fictional_reference(*, config, artifact_writer, implementation_commit):
        nonlocal cached_reference
        if cached_reference is None:
            parent, _ = selected_events.read_selected_parent(config=request, artifact_writer=artifact_writer, implementation_commit=implementation_commit)
            cached_reference = _reference_report(SimpleNamespace(null_reference=config, source=config.source), parent)
        return {"report": cached_reference}
    monkeypatch.setattr(selected_events, "read_matrix_geometry_result", fictional_reference)
    _selected_view_sensitive_public_case(config, plane, monkeypatch)
    assert len(calls) == 1
    assert calls[0].scientific_seed_sha256 == selected_events.NOMINATED_PARENT_DRIVER


def _selected_view_sensitive_public_case(bound, plane, monkeypatch):
    """Reuse the complete software reference; a manufactured numerical stop is explicit."""
    from empirical_lawhood.api import selected_events
    from empirical_lawhood.adapters.methods.matrix_response_study import shooting_committor
    request = replace(selected_events.selected_parent_example_input(environment_lock_sha256=bound.environment_lock_sha256), minimum_free_bytes=1)
    selected_events.reconstruct_nominated_selected_parent(config=request, artifact_writer=plane, project_root=ROOT, implementation_commit="0" * 40)
    config = selected_events.prepare_selected_event(config_id="synthetic.full-selected-negative",
        parent_request=request, null_reference=bound, master_seed=992747, artifact_writer=plane, implementation_commit="0" * 40)
    original_bridge = shooting_committor.execute_brownian_bridge_replay
    def manufactured_stop(**kwargs):
        actual = original_bridge(**kwargs)
        return replace(actual, disposition=shooting_committor.MatrixResponseShootingCommittorNumericalDisposition.EVENT_NUMERICAL_VIEW_SENSITIVE,
            reason_codes=tuple(sorted(set((*actual.reason_codes, "synthetic-manufactured-view-sensitivity")))))
    monkeypatch.setattr(shooting_committor, "execute_brownian_bridge_replay", manufactured_stop)
    monkeypatch.setattr(shooting_committor, "_run_branch_task", lambda *_: pytest.fail("view-sensitive path entered conditional future"))
    result = selected_events.run_selected_event(config=config, artifact_writer=plane, project_root=ROOT, implementation_commit="0" * 40)
    assert result["report"].scientific_terminal == "EVENT_NUMERICAL_VIEW_SENSITIVE"
    assert set(result["report"].condition_false_descendants) >= {"shooting", "local-linearization"}
    monkeypatch.setattr(shooting_committor, "execute_brownian_bridge_replay", lambda **_: pytest.fail("read repeated bridge"))
    fresh = ExternalArtifactPlane(plane.root)
    recovered = selected_events.read_selected_event_result(config=config, artifact_writer=fresh, implementation_commit="0" * 40)
    assert recovered["report"] == result["report"] and not recovered["cohorts"]
