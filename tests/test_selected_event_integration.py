"""Current selected conditioning, genuine bounded native seams and independent falsifiers."""

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from empirical_lawhood.api import selected_events as api
from empirical_lawhood.api import matrix_geometry as geometry_api
from empirical_lawhood.api.matrix_native_custody import MatrixNativeCustody
from empirical_lawhood.adapters.methods.matrix_response_study import shooting_committor as science
from empirical_lawhood.adapters.methods.matrix_response_study import anisotropic_feasibility as geometry
from empirical_lawhood.adapters.simulators.six_matrix_response import shooting as native
from empirical_lawhood.adapters.simulators.six_matrix_response.model import ideal_state
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import SixMatrixResponseConstitution, SixMatrixResponseFastReceiver
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.artifacts import ExternalRootContract
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    store = tmp_path_factory.mktemp("synthetic-selected-event")
    plane = ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract(
        "synthetic.selected-store", "software exposed conditioning", str(store), "/", OperatorStorageProfile.SCHEMA, 1, None, None, ())))
    lock = sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    request = replace(api.selected_parent_example_input(environment_lock_sha256=lock), minimum_free_bytes=1)
    parent, manifest = api.reconstruct_nominated_selected_parent(config=request, artifact_writer=plane, project_root=ROOT, implementation_commit="0" * 40)
    null = replace(geometry_api.matrix_geometry_example_input(environment_lock_sha256=lock), minimum_free_bytes=1)
    config = api.prepare_selected_event(config_id="synthetic.selected-event", parent_request=request,
        null_reference=null, master_seed=87348, artifact_writer=plane, implementation_commit="0" * 40)
    return plane, request, parent, manifest, config


def test_actual_nominated_parent_fixed_context_and_complete_current_noncontact_proof(scenario, monkeypatch):
    plane, request, parent, _, config = scenario
    monkeypatch.setattr(native, "replay_selected_co_anneal_event", lambda **_: pytest.fail("proof entered replay"))
    proof = api.prove_selected_event(config=config, artifact_writer=plane, implementation_commit="0" * 40)
    assert parent.rollout.final_state_sha256 == api.NOMINATED_PARENT_FINAL_STATE
    assert parent.rollout.rng_seed_sha256 == api.NOMINATED_PARENT_DRIVER
    assert proof["independent_selected_events"] == 1 and proof["conditional_futures"] == 448
    assert proof["checkpoint_steps"] == (816, 848, 880, 896, 928, 960, 1024)
    assert proof["maximum_integration_steps"] == 461952
    assert not proof["scientific_execution_performed"]
    assert decode_canonical_bytes(config.canonical_bytes(), type(config), maximum_bytes=api.SELECTED_EVENT_INPUT_MAXIMUM_BYTES) == config
    assert api.reconstruct_nominated_selected_parent(config=request, artifact_writer=plane, project_root=ROOT, implementation_commit="0" * 40)[0] == parent


def test_branch_roster_effective_stream_exclusion_and_parent_identity_refuse(scenario):
    _, _, parent, _, config = scenario
    with pytest.raises(ValueError, match="seven complete"):
        replace(config, branch_seeds=config.branch_seeds[:-1])
    with pytest.raises(ValueError, match="bridge allocation"):
        replace(config, bridge_seeds=config.bridge_seeds[:-1])
    with pytest.raises(ValueError, match="effective"):
        replace(config, branch_seeds=(replace(config.branch_seeds[0], full_seed_sha256=config.branch_seeds[1].full_seed_sha256[:32] + "a" * 32), *config.branch_seeds[1:]))
    with pytest.raises(ValueError, match="exposed parent"):
        replace(config, branch_seeds=(replace(config.branch_seeds[0], full_seed_sha256=api.NOMINATED_PARENT_DRIVER[:32] + "a" * 32), *config.branch_seeds[1:]))
    with pytest.raises(ValueError, match="no admitted"):
        replace(parent, rollout=replace(parent.rollout, constitution=SixMatrixResponseConstitution.EMPTY,
            factor_y=replace(parent.rollout.factor_y, geometric=False, reason_codes=("synthetic-no-event",))))
    with pytest.raises(ValueError, match="cannot be replaced"):
        replace(parent, rollout=replace(parent.rollout, rng_seed_sha256="a" * 64))


def test_meaningful_new_future_allocation_changes_seeds_and_labels_do_not(scenario):
    plane, request, _, _, config = scenario
    same = api.prepare_selected_event(config_id="synthetic.renamed", parent_request=request,
        null_reference=config.null_reference, master_seed=87348, artifact_writer=plane, implementation_commit="0" * 40)
    other = api.prepare_selected_event(config_id="synthetic.changed-driver", parent_request=request,
        null_reference=config.null_reference, master_seed=87349, artifact_writer=plane, implementation_commit="0" * 40)
    assert same.branch_seeds == config.branch_seeds
    assert other.branch_seeds != config.branch_seeds
    assert same.run_id != config.run_id


def test_actual_public_replay_restart_future_retention_and_fresh_recovery(scenario, monkeypatch):
    plane, _, _, _, config = scenario
    kwargs = dict(config=config, checkpoint_step=896, branch_index=0, artifact_writer=plane,
        project_root=ROOT, implementation_commit="0" * 40)
    result = api.run_selected_event_conformance(**kwargs)
    assert result["replay"].parent_replay_exact and result["replay"].checkpoint_restart_exact
    assert result["branch"].checkpoint_step == 896 and result["branch"].branch_index == 0
    assert result["branch"].seed_document_sha256 == next(s.full_seed_sha256 for s in config.branch_seeds if (s.checkpoint_step, s.stream_index) == (896, 0))
    assert not result["full_investigation_performed"] and not result["event_qualification_performed"]
    monkeypatch.setattr(native, "replay_selected_co_anneal_event", lambda **_: pytest.fail("completed replay repeated"))
    monkeypatch.setattr(science, "_run_branch_task", lambda *_: pytest.fail("completed future repeated"))
    fresh = ExternalArtifactPlane(plane.root)
    assert api.run_selected_event_conformance(**{**kwargs, "artifact_writer": fresh}) == result
    wrong = replace(config, protocol=replace(config.protocol, null_phi_target=config.protocol.null_phi_target + Decimal("0.01")))
    with pytest.raises(ValueError, match="mathematical protocol"):
        api.run_selected_event_conformance(**{**kwargs, "config": wrong})


def _reference_report(config, parent):
    """Complete fictional reference census for reducer conformance, not native evidence."""
    tasks = tuple(t for t in geometry_api._all_tasks(config.null_reference) if t.stage is geometry.MatrixResponseAnisotropicFeasibilityStage.FEASIBILITY)
    rows = []
    for task in tasks:
        row = replace(parent.rollout, rollout_id=geometry._rollout_id(task), member_id=task.member.member_id,
            alpha_x_index=task.alpha_x_index, alpha_y_index=task.alpha_y_index,
            alpha_tilde_x=task.alpha_tilde_x, alpha_tilde_y=task.alpha_tilde_y,
            history_id=task.history_id, seed_index=task.seed_index, rng_seed_sha256=task.scientific_seed_sha256)
        rows.append(row)
    return geometry.MatrixResponseAnisotropicFeasibilityQualificationReport("synthetic.reference-report", "0" * 40,
        "0" * 64, ObjectIdentity.from_record(config.source.config_id, config.source), tuple(sorted(rows, key=lambda r: r.rollout_id)),
        (), (), None, (), geometry.MatrixResponseAnisotropicFeasibilityDisposition.NO_CONSTITUTIVE_SUBSTRATE,
        ("synthetic-no-recurring-candidate",), EvidenceCeiling.NON_PROMOTABLE, OutcomeAccess.DEVELOPMENT_VISIBLE, VisibilityCeiling.DEVELOPMENT_ONLY, False)


def test_new_measured_null_counts_remain_valid_and_incomplete_census_is_unevaluable(scenario):
    _, _, parent, _, config = scenario
    report = _reference_report(config, parent)
    result = science.execute_amplitude_conditioned_null(report=report, config=config.protocol, selected_parent=parent.rollout)
    assert result.reference_inventory_complete
    assert result.disposition is science.MatrixResponseShootingCommittorNullDisposition.ALGEBRAIC_ORGANIZATION_NOT_EXCEPTIONAL
    # A fresh trajectory at the nominated coordinate is not the exposed parent.
    assert result.eligible_reference_count == 6 * 13 * 12 * 3 * 3
    primary = next(c for c in result.cells if c.amplitude_band == Decimal("0.01") and c.stratum_id == "all-c1a")
    assert primary.denominator != 588 and primary.joint_count > 0
    partial = science.execute_amplitude_conditioned_null(report=replace(report, rollouts=report.rollouts[:-1]), config=config.protocol, selected_parent=parent.rollout)
    assert not partial.reference_inventory_complete
    assert partial.disposition is science.MatrixResponseShootingCommittorNullDisposition.AMPLITUDE_NULL_UNEVALUABLE


def test_ou_bridge_independent_composition_and_empirical_marginals():
    rng = np.random.Generator(np.random.PCG64DXSM(483848))
    coarse = rng.standard_normal(20000).astype(np.complex128).reshape((20000, 1, 1))
    zeta = rng.standard_normal(20000).astype(np.complex128).reshape((20000, 1, 1))
    decay = np.exp(-0.001 / 2)
    first, second = native.brownian_bridge_split(coarse_noise=coarse, bridge_noise=zeta, half_decay=decay)
    assert np.allclose((decay * first + second) / np.sqrt(1 + decay**2), coarse, atol=5e-16)
    assert np.var(first.real) == pytest.approx(1, abs=0.035)
    assert np.var(second.real) == pytest.approx(1, abs=0.035)
    assert np.corrcoef(first.real.ravel(), second.real.ravel())[0, 1] == pytest.approx(0, abs=0.03)


def _observation(index, config, *, kernel="0.1"):
    step = 16 * (index + 1)
    zero = Decimal(0)
    receiver = SixMatrixResponseFastReceiver(f"synthetic.receiver-{step}", step, config.target_alpha_tilde_x, config.target_alpha_tilde_y,
        zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, True, True, ())
    x = science.MatrixResponseShootingCommittorFactorObservation("X", Decimal("0.6"), Decimal("0.1"), Decimal("0.1"), True, False, True)
    y = science.MatrixResponseShootingCommittorFactorObservation("Y", Decimal("0.6"), Decimal("0.1"), Decimal(kernel), True, Decimal(kernel) <= Decimal("0.25"), True)
    return science.MatrixResponseShootingCommittorObservation(f"synthetic.sample-{step}", step, Decimal(step) / 1000, receiver, x, y, True)


def test_forward_only_rolling_window_kernel_sensitivity_and_independent_survival(scenario):
    config = scenario[-1].protocol
    observations = tuple(_observation(i, config) for i in range(64))
    labels = science.rolling_labels(observations, config=config)
    assert len(labels) == 49 and labels[0].endpoint_step == 256 and labels[0].label == "01"
    altered = (_observation(0, config, kernel="0.3"), *observations[1:])
    sensitive = science.rolling_labels(altered, config=config)
    assert sensitive[0].label == "01" and not sensitive[0].strict_window_y_geometric
    assert sensitive[1].strict_window_y_geometric
    curve = science._kaplan_meier(((0.25, True), (0.5, False), (0.75, True), (1, False)))
    assert [p.at_risk for p in curve] == [4, 3, 2, 1]
    assert science._rmst(curve, 1) == pytest.approx(0.71875)
    for success in (0, 32, 64):
        interval = science.wilson_interval(success, 64, confidence_level=0.95)
        assert interval.lower <= Decimal(success) / 64 <= interval.upper


def test_linearization_independent_quadratic_and_gauge_basis():
    hessian = np.diag(np.asarray((-2, 1, 3), dtype=np.float64))
    observed = science._central_jacobian(lambda x: hessian @ x, np.ones(3), 1e-6)
    assert np.allclose(observed, hessian, atol=2e-10)
    basis = native.traceless_hermitian_basis(4)
    assert basis.shape == (15, 4, 4)
    gram = np.asarray([[np.vdot(left, right).real for right in basis] for left in basis])
    assert np.allclose(gram, np.eye(15), atol=1e-14)
    assert np.allclose(np.trace(basis, axis1=-2, axis2=-1), 0, atol=1e-14)
    for signs, expected in (((1, 1, 1), 0), ((-1, 1, 2), 1), ((-2, -1, 2), 2)):
        hessian = np.diag(np.asarray(signs, dtype=float))
        spectrum = science._drift_spectrum(hessian=hessian, raw_hessian=hessian,
            step_size=1e-6, quotient_basis=np.eye(6), quotient_rank=0, unstable_cutoff=1e-6)
        assert len(spectrum.scale.unstable_eigenvalues) == expected


def _current_conformance(scenario):
    return api.run_selected_event_conformance(config=scenario[-1], checkpoint_step=896, branch_index=0,
        artifact_writer=scenario[0], project_root=ROOT, implementation_commit="0" * 40)


def test_complete_64_future_terminal_denominators_no_top_up_and_missing_foreign_refusals(scenario):
    """Engineered terminal labels over one real branch, solely reducer conformance."""
    result = _current_conformance(scenario)
    config = scenario[-1].protocol
    checkpoint = result["replay"].checkpoints[3]
    base = tuple(replace(result["branch"], branch_id=f"synthetic.branch-{i:02d}", branch_index=i) for i in range(64))
    terminals = science.MatrixResponseShootingCommittorBranchTerminal
    def terminalized(terminal):
        return tuple(replace(branch, terminal=terminal,
            tau_g=Decimal("0.256") if terminal is terminals.G_FIRST_WITHIN_HORIZON else None,
            tau_00=Decimal("0.5") if terminal is terminals.ROBUST_00_FIRST_WITHIN_HORIZON else None,
            tau_00_after_g=None,
            reason_codes=("synthetic-invalid",) if terminal is terminals.INVALID_NUMERICAL_FUTURE else ()) for branch in base)
    for terminal in terminals:
        cohort = science.reduce_shooting_cohort(checkpoint=checkpoint, branches=terminalized(terminal), config=config)
        assert cohort.unconditional_committor.denominator == 64
        assert sum((cohort.g_first_count, cohort.robust_00_first_count, cohort.right_censored_count, cohort.invalid_count)) == 64
        if terminal is terminals.G_FIRST_WITHIN_HORIZON:
            assert cohort.unconditional_committor.estimate == 1 and cohort.landmark_branch_count == 64
        if terminal is terminals.RIGHT_CENSORED_NO_TARGET_WITHIN_HORIZON:
            assert cohort.resolved_committor.denominator == 0
        if terminal is terminals.INVALID_NUMERICAL_FUTURE:
            assert not cohort.complete and cohort.valid_branch_count == 0
    for branches in (base[:-1], (base[0], *base[:-1])):
        with pytest.raises(ValueError, match="roster"):
            science.reduce_shooting_cohort(checkpoint=checkpoint, branches=branches, config=config)
    with pytest.raises(ValueError, match="foreign"):
        science.reduce_shooting_cohort(checkpoint=checkpoint,
            branches=(replace(base[0], checkpoint_id="synthetic.foreign-checkpoint"), *base[1:]), config=config)


def test_actual_same_driver_bridge_retains_view_disposition_and_bad_allocation_refuses_before_step(scenario, monkeypatch):
    plane, _, parent, _, config = scenario
    result = _current_conformance(scenario)
    custody = api._custody(config, plane, "0" * 40)
    transport = custody.load("conformance-replay", api.MatrixSelectedReplayTransport)
    exported = custody.store.read_by_receipt_id(config.run_id, "conformance-seed-export",
        f"receipt.{config.run_id}.conformance-seed-export.attempt-001")
    seeds = tuple(api._typed_seed(seed, role="shooting-bridge", root_id=config.protocol.parent_rollout_id,
        context=config.protocol.fingerprint(), source=ObjectIdentity.from_record(config.config_id, config),
        receipt=ObjectIdentity.from_record(exported.receipt_id, exported)) for seed in config.bridge_seeds)
    member = next(m for m in config.source.anisotropic_model.family_members if m.member_id == config.protocol.member_id)
    kwargs = dict(replay=transport.replay(), replay_result=result["replay"], member=member,
        primary_view=config.source.primary_view, half_view=config.source.secondary_view, config=config.protocol)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        bridge = science.execute_brownian_bridge_replay(**kwargs, scientific_seed_inputs=seeds)
    assert bridge.half_step_count == 2048 and len(bridge.alignments) == 7
    assert bridge.bridge_seed_aggregate_sha256 == sha256("".join(s.full_seed_sha256 for s in config.bridge_seeds).encode()).hexdigest()
    custody.retain("conformance-bridge", bridge)
    assert api._custody(config, ExternalArtifactPlane(plane.root), "0" * 40, read_only=True).load("conformance-bridge", type(bridge)) == bridge
    monkeypatch.setattr(science, "baoab_step_with_hermitian_noise", lambda *_a, **_k: pytest.fail("incomplete bridge entered numerical step"))
    with pytest.raises(ValueError, match="roster|complete|census"):
        science.execute_brownian_bridge_replay(**kwargs, scientific_seed_inputs=seeds[:-1])
    # Favorable descriptive null cannot compensate a view-sensitive path.
    null = science.execute_amplitude_conditioned_null(report=_reference_report(config, parent), config=config.protocol, selected_parent=parent.rollout)
    favorable = replace(null, disposition=science.MatrixResponseShootingCommittorNullDisposition.ALGEBRAIC_ORGANIZATION_EXCEPTIONAL_AT_MATCHED_AMPLITUDE)
    sensitive = replace(bridge, disposition=science.MatrixResponseShootingCommittorNumericalDisposition.EVENT_NUMERICAL_VIEW_SENSITIVE,
        reason_codes=("synthetic-view-sensitive",))
    terminal = science.finalize_matrix_response_study_shooting_committor(implementation_commit="0" * 40,
        config=config.protocol, replay=result["replay"], bridge=sensitive, cohorts=(), linearization=None, amplitude_null=favorable)
    assert terminal.scientific_terminal == "EVENT_NUMERICAL_VIEW_SENSITIVE"
    assert "shooting" in terminal.condition_false_descendants and not terminal.grants_authority


def test_actual_engineered_linearization_preserves_gauge_and_nondifferentiable_receivers(scenario):
    config = scenario[-1].protocol
    member = next(m for m in scenario[-1].source.anisotropic_model.family_members if m.member_id == config.member_id)
    state = replace(ideal_state(q=2, alpha_tilde_x=float(config.target_alpha_tilde_x),
        alpha_tilde_y=float(config.target_alpha_tilde_y), constitution="01"), step_index=896)
    checkpoint = native.checkpoint_from_replay_state(parent_rollout_id=config.parent_rollout_id,
        parent_report_sha256=config.parent_anisotropic_feasibility_report_sha256, parent_time=Decimal("0.896"),
        state=state, rng=np.random.Generator(np.random.PCG64DXSM(817)), fast_receiver_prefix=(), spectral_receiver_prefix=())
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result = science.execute_local_linearization(checkpoint=checkpoint, member=member, config=config)
    assert result.position_dimension == 96 and result.phase_space_dimension == 192 and result.quotient_tangent_rank == 12
    derivatives = {d.receiver_id.rsplit(".", 1)[-1]: d for d in result.receiver_derivatives}
    assert derivatives["y-radius"].differentiable
    for name in ("y-closure", "y-kernel", "cross-commutator"):
        assert not derivatives[name].differentiable
        assert derivatives[name].reason_codes == ("active-constituent-gap-insufficient",)


def test_wrong_source_and_interrupted_closed_inventory_refuse_before_native(scenario, monkeypatch):
    plane, _, _, _, config = scenario
    monkeypatch.setattr(native, "replay_selected_co_anneal_event", lambda **_: pytest.fail("invalid input entered replay"))
    with pytest.raises(ValueError, match="numerical source preset"):
        replace(config, parent_request=replace(config.parent_request, source=replace(config.source, config_id="synthetic.other-source")))
    with pytest.raises(ValueError, match="no complete terminal"):
        api.read_selected_event_result(config=config, artifact_writer=plane, implementation_commit="0" * 40)
    custody = MatrixNativeCustody(writer=plane, config=config, run_id=config.run_id,
        implementation_commit="0" * 40, minimum_free_bytes=1, maximum_output_bytes=config.maximum_output_bytes)
    custody.begin_native_effect("bridge")
    with pytest.raises(ValueError, match="uncertain effects"):
        custody.load("bridge", science.MatrixResponseShootingCommittorBridgeResult)
