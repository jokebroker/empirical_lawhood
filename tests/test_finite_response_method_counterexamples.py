"""Independent finite prospective evaluation/preparation-policy development joins on explicitly synthetic old-name operands."""

from tests.finite_response_seed_fixtures import ASSIGNED_SEEDS
from dataclasses import replace
from decimal import Decimal as D
from hashlib import sha256

import numpy as np
import pytest

from empirical_lawhood.adapters.methods.finite_response_law.assigned_prediction import AssignedPrediction
from empirical_lawhood.adapters.methods.finite_response_law.calibration_analysis import calibration_scores
from empirical_lawhood.adapters.methods.finite_response_law.calibration_panel import CalibrationPanel
from empirical_lawhood.adapters.methods.finite_response_law.consumer import consumer_task, evaluation_requests, response_coordinates
from empirical_lawhood.adapters.methods.finite_response_law.evaluation_readout import BOUNDARIES, FiniteResponseLawRootInferenceOperands, FiniteResponseLawScalarConsumerEvent, cohort_inference
from empirical_lawhood.adapters.methods.finite_response_law.independent_calibration import independent_scores
from empirical_lawhood.adapters.methods.finite_response_law.intervals import choose
from empirical_lawhood.adapters.methods.finite_response_law.science import seed_for
from empirical_lawhood.adapters.methods.finite_response_law.preparation_policy_fitting import PreparationPolicyFeatures, PreparationPolicyPanel, development_gates, provisional_quantile
from empirical_lawhood.adapters.methods.finite_response_law.preparation_policy_screen import actual_joint, full_menu_adequacy
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedEvaluationRoot
from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_contracts import FiniteResponseLawEvaluationRoot
from empirical_lawhood.kernel.provenance import ObjectIdentity


class _ZeroLower:
    q = D(0)

    def predict(self, native_handoff: np.ndarray) -> AssignedPrediction:
        n = len(native_handoff)
        return AssignedPrediction(
            np.zeros((n, 4, 8), dtype=np.float64),
            np.full((n, 4, 8), 1e-8, dtype=np.float64),
            np.ones(n, dtype=np.bool_),
        )


def test_informative_composition_truth_blind_interval_choice_and_uncertainty_refusal() -> None:
    prediction = np.zeros((1, 5, 4, 8), dtype=np.float64)
    prediction[:, :, 0, 0] = 0.04
    sigma = np.full_like(prediction, 1e-6)
    support = np.ones((1, 5), dtype=np.bool_)
    direction = np.zeros((1, 256, 2), dtype=np.int64)
    requirement = np.full(direction.shape, 0.03, dtype=np.float64)
    admitted = choose(prediction, sigma, 1.0, support, direction, requirement)
    assert np.all(admitted.selected == 1)  # positive 8-unit first direction
    assert not admitted.feasible[..., 0].any()  # negative sign fails both requests
    assert not (admitted.halfwidth > 0.01).any()
    uncertain = choose(prediction, sigma, float("inf"), support, direction, requirement)
    assert np.all(uncertain.selected == -1)
    unsupported = choose(
        prediction, sigma, 1.0, np.zeros_like(support), direction, requirement
    )
    assert np.all(unsupported.selected == -1)


def _preparation_policy_features() -> PreparationPolicyFeatures:
    roots = tuple(f"prepared-response.prepared.r{i:03d}" for i in range(8)) + tuple(
        f"information-response-prediction.prepared.r{i:03d}" for i in range(16)
    )
    return PreparationPolicyFeatures(
        roots,
        tuple("prepared-response" if i < 8 else "information-response-prediction" for i in range(24)),
        np.r_[np.arange(8), np.arange(16)].astype(np.int64),
        np.zeros((24, 24, 2), dtype=np.float64),
        np.zeros((24, 9, 24, 2), dtype=np.float64),
    )


def test_preparation_screen_all_nine_schedule_maximum_and_two_consumer_joint() -> None:
    y = np.zeros((24, 9, 4, 8, 2, 2), dtype=np.float64)
    measured = np.ones_like(y, dtype=np.bool_)
    work = np.zeros((24, 9, 2), dtype=np.float64)
    panel = PreparationPolicyPanel(y, measured, measured.copy())
    baseline = full_menu_adequacy(_ZeroLower(), _preparation_policy_features(), panel, work)
    assert baseline.event.shape == (24, 9)
    assert baseline.event.all()
    late = y.copy()
    late[0, 8, 0, 0, 1, 1] = 0.02
    changed = full_menu_adequacy(
        _ZeroLower(),
        _preparation_policy_features(),
        PreparationPolicyPanel(late, measured, measured.copy()),
        work,
    )
    assert not changed.event[0, 8]
    assert changed.event[0, :8].all()
    assert changed.event[1:].all()
    assert changed.maximum_error[0, 8] >= 2

    response = y.copy()
    response[:, :, 0, 0] = 0.04
    response_panel = PreparationPolicyPanel(response, measured, measured.copy())
    roots = np.arange(24, dtype=np.int64)
    schedules = np.zeros(24, dtype=np.int64)
    selected = np.ones((24, 256, 2), dtype=np.int64)
    directions = np.zeros_like(selected)
    requirement = np.full(selected.shape, 0.03, dtype=np.float64)
    assert actual_joint(
        response_panel, work, roots, schedules, selected, directions, requirement
    ).all()
    requirement[:, :, 1] = 0.06
    assert not actual_joint(
        response_panel, work, roots, schedules, selected, directions, requirement
    ).any()


def test_preparation_screen_rank_23_infinity_and_three_independent_negative_gates() -> None:
    assert provisional_quantile(np.arange(18, dtype=np.float64), expected_n=18) == (
        18,
        17,
    )
    rank18, q18 = provisional_quantile(
        np.r_[np.arange(17, dtype=np.float64), np.inf], expected_n=18
    )
    assert rank18 == 18 and np.isinf(q18)
    assert provisional_quantile(np.arange(24, dtype=np.float64), expected_n=24) == (
        23,
        22,
    )
    rank, value = provisional_quantile(
        np.r_[np.arange(22, dtype=np.float64), np.inf, np.inf], expected_n=24
    )
    assert rank == 23 and np.isinf(value)
    adequacy = np.zeros((24, 9), dtype=np.bool_)
    adequacy[:6, 1] = True
    selected = np.zeros(24, dtype=np.int64)
    selected[:6] = 1
    fixed = np.zeros(24, dtype=np.int64)
    selected_joint = np.zeros((24, 256), dtype=np.bool_)
    selected_joint.flat[:400] = True
    fixed_joint = np.zeros_like(selected_joint)
    passed = development_gates(adequacy, selected, fixed, selected_joint, fixed_joint)
    assert (passed.headroom_roots, passed.adequacy_net_roots) == (6, 6)
    assert passed.joint_improvement > 0.05 and passed.passed
    too_few_headroom = adequacy.copy()
    too_few_headroom[5, 1] = False
    assert not development_gates(
        too_few_headroom, selected, fixed, selected_joint, fixed_joint
    ).passed
    too_few_net = selected.copy()
    too_few_net[3:6] = 0
    assert not development_gates(
        adequacy, too_few_net, fixed, selected_joint, fixed_joint
    ).passed
    too_little_joint = selected_joint.copy()
    too_little_joint.flat[300:] = False
    assert not development_gates(
        adequacy, selected, fixed, too_little_joint, fixed_joint
    ).passed


def _prospective_evaluation_root(index: int) -> FiniteResponseLawRootInferenceOperands:
    events = tuple(
        FiniteResponseLawScalarConsumerEvent(
            f"{boundary}.consumer-{consumer}", 1, 1, True, False, True, True
        )
        for boundary in sorted(BOUNDARIES)
        for consumer in (0, 1)
    )
    return FiniteResponseLawRootInferenceOperands(
        f"prospective-evaluation.r{index:03d}",
        events,
        tuple((b, D("0.9") if b == "composed" else D(1)) for b in BOUNDARIES),
        True,
        tuple((b, (D(0),) * 32) for b in BOUNDARIES),
    )


def test_prospective_evaluation_joint_use_false_admission_and_complete_root_denominator() -> None:
    roots = tuple(_prospective_evaluation_root(i) for i in range(64))
    clear = cohort_inference(roots, lower_qualified=True, cached_qualified=True)
    assert clear["independent_roots"] == 64
    assert clear["joint_successes"] == 64
    assert clear["false_admission_episodes"] == 0
    assert clear["tier1_use_supported"]
    assert clear["tier1_information_supported"]
    assert not clear["tier1_added_use_supported"]  # no paired discordance

    assigned_namespace = "empirical-lawhood.finite-response-law.prospective-evaluation.development"
    assigned = tuple(
        replace(root, root_id=f"{assigned_namespace}.r{index:03d}", bootstrap_seed=next(seed for p, i, seed in ASSIGNED_SEEDS[assigned_namespace] if p == "bootstrap" and i == -1))
        for index, root in enumerate(roots)
    )
    assigned_readout = cohort_inference(
        assigned, lower_qualified=True, cached_qualified=True
    )
    assert assigned_readout["independent_roots"] == 64
    assert assigned_readout["joint_successes"] == clear["joint_successes"]
    with pytest.raises(ValueError, match="every assigned root"):
        cohort_inference(
            (*assigned[:-1], roots[-1]), lower_qualified=True, cached_qualified=True
        )

    adverse = list(roots)
    for index in (0, 1, 2):
        first = next(
            e for e in adverse[index].events if e.policy_id == "composed.consumer-0"
        )
        events = tuple(
            replace(e, success=False, false_admission=True) if e == first else e
            for e in adverse[index].events
        )
        adverse[index] = replace(adverse[index], events=events)
    refused = cohort_inference(
        tuple(adverse), lower_qualified=True, cached_qualified=True
    )
    assert refused["joint_successes"] == 61
    assert refused["false_admission_episodes"] == 3
    assert not refused["tier1_use_supported"]
    assert not refused["preparation_policy_eligible"]


def test_calibration_calibration_keeps_failed_roots_in_rank_30() -> None:
    shape = (32, 1, 4, 8, 2, 2)
    y = np.zeros(shape, dtype=np.float64)
    observed = np.ones(shape, dtype=np.bool_)
    valid = observed.copy()
    features = np.zeros((32, 24, 2), dtype=np.float64)
    work = np.zeros((32, 1, 2), dtype=np.float64)
    parents = np.zeros(32, dtype=np.int64)
    roots = tuple(f"calibration.r{i:03d}" for i in range(32))
    prediction = AssignedPrediction(
        np.zeros((32, 4, 8), dtype=np.float64),
        np.ones((32, 4, 8), dtype=np.float64),
        np.ones(32, dtype=np.bool_),
    )
    for failed_count, q_is_infinite in ((2, False), (3, True)):
        observed[:failed_count] = False
        valid[:failed_count] = False
        panel = CalibrationPanel(
            y, observed, valid, features, features, work, parents, roots
        )
        actual = calibration_scores(panel, prediction)
        independent, independent_q, reasons = independent_scores(
            panel, prediction.mean, prediction.sigma, prediction.supported
        )
        assert actual.rank == 30
        assert np.isinf(actual.q) == q_is_infinite
        np.testing.assert_array_equal(actual.root_scores, independent)
        assert actual.q == independent_q
        assert all("MEASUREMENT_UNAVAILABLE" in reasons[i] for i in range(failed_count))
        assert len(actual.root_scores) == 32


def test_prospective_evaluation_two_consumer_requests_commit_their_exact_target_bytes() -> None:
    old_root = FiniteResponseLawEvaluationRoot(
        "prospective-evaluation",
        0,
        sha256(str(seed_for("prefix", "prospective-evaluation.r000")).encode()).hexdigest(),
        None,
    )
    first, second = evaluation_requests(old_root)
    assert (first, second) == evaluation_requests(old_root)
    assert first.root_id == second.root_id == "prospective-evaluation.r000"
    assert (first.consumer, second.consumer) == (0, 1)
    assert first.request_id != second.request_id
    second_root = FiniteResponseLawEvaluationRoot(
        "prospective-evaluation",
        1,
        sha256(str(seed_for("prefix", "prospective-evaluation.r001")).encode()).hexdigest(),
        None,
    )
    assert evaluation_requests(second_root)[0].root_id != first.root_id
    readout = ObjectIdentity(
        "synthetic-prospective-evaluation-readout",
        'empirical-lawhood/test/synthetic-readout',
        "1.0.0",
        "0" * 64,
    )
    coordinates = response_coordinates(readout)
    task = consumer_task(first, coordinates)
    assert len(coordinates) == 16  # two futures, eight quantities; one root
    assert task.target_reveal == ObjectIdentity.from_record(first.request_id, first)
    changed = replace(first, lower=first.lower + D("0.00001"))
    assert consumer_task(changed, coordinates).target_reveal != task.target_reveal
    assigned_namespace = "empirical-lawhood.finite-response-law.prospective-evaluation.development"
    assigned_unit = f"{assigned_namespace}.r000"
    assigned = FiniteResponseLawAssignedEvaluationRoot(
        "prospective-evaluation",
        0,
        sha256(str(next(seed for p, i, seed in ASSIGNED_SEEDS[assigned_namespace] if p == "prefix" and i == 0)).encode()).hexdigest(),
        None,
        assigned_namespace,
    scientific_seeds=tuple((p, s) for p, i, s in ASSIGNED_SEEDS[assigned_namespace] if i in (0, -1)),
    )
    assigned_pair = evaluation_requests(assigned)
    assert assigned_pair == evaluation_requests(assigned)
    assert tuple(r.root_id for r in assigned_pair) == (assigned_unit, assigned_unit)
    assert tuple(r.request_id for r in assigned_pair) != (first.request_id, second.request_id)
    with pytest.raises(TypeError, match="validated evaluation root"):
        evaluation_requests(0)
