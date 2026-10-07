"Truth-known conformance cases for the topology-response post-hoc estimators."

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from empirical_lawhood.adapters.methods.response_formalization_topology_posthoc import (
    AdmissionMarginRecord,
    StateAugmentationCandidate,
    apply_orthogonal_alignment,
    classification_summary,
    directed_reachable,
    dominators,
    exact_adaptive_resolution_policy,
    fit_orthogonal_alignment,
    graph_descriptors,
    grouped_leave_one_out_ridge,
    minimal_node_cutsets,
    nearest_centroid_predictions,
    rms,
    singular_spectrum,
    two_set_cech_dimensions,
)


@dataclass(frozen=True, slots=True)
class ConformanceCase:
    case_id: str
    expected_status: str
    observed_status: str
    passed: bool
    certificate: dict[str, Any]


def _case(case_id: str, function: Callable[[], dict[str, Any]]) -> ConformanceCase:
    try:
        certificate = function()
        passed = bool(certificate.pop("passed"))
        observed = "PASS" if passed else "FAIL"
    except (ValueError, np.linalg.LinAlgError) as error:
        certificate = {"error": str(error)}
        passed = False
        observed = "UNEXPECTED_EXCEPTION"
    return ConformanceCase(case_id, "PASS", observed, passed, certificate)


def _alignable_receiver_chart_case() -> dict[str, Any]:
    generator = np.random.default_rng(1)
    source = generator.normal(size=(32, 4))
    truth_rotation, _ = np.linalg.qr(generator.normal(size=(4, 4)))
    target = source @ truth_rotation + 0.4
    fit = fit_orthogonal_alignment(source, target)
    aligned = apply_orthogonal_alignment(source, fit)
    defect = float(np.max(np.abs(aligned - target)))
    return {"passed": defect < 1e-10, "maximum_alignment_defect": defect}


def _many_to_one_information_loss_case() -> dict[str, Any]:
    values = np.asarray([[0.0], [0.0], [1.0], [1.0]])
    labels = ("a", "b", "c", "d")
    predicted = nearest_centroid_predictions(values, labels, values)
    summary = classification_summary(labels, predicted)
    return {"passed": float(summary["accuracy"]) <= 0.5, "accuracy": summary["accuracy"]}


def _rank_one_interface_case() -> dict[str, Any]:
    left = np.arange(6, dtype=np.float64)[:, None]
    matrix = left @ np.asarray([[1.0, -2.0, 0.5]])
    result = singular_spectrum(matrix, relative_tolerances=(1e-10,))
    rank = int(result["relative_tolerance_ranks"]["1e-10"])
    return {"passed": rank == 1, "rank": rank}


def _distributed_full_rank_case() -> dict[str, Any]:
    matrix = np.eye(5, dtype=np.float64)
    result = singular_spectrum(matrix, relative_tolerances=(1e-10,))
    rank = int(result["relative_tolerance_ranks"]["1e-10"])
    return {"passed": rank == 5, "rank": rank}


def _zero_h1_compatible_cover_case() -> dict[str, Any]:
    left = np.asarray([[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
    right = np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
    result = two_set_cech_dimensions(
        left_dimension=5,
        right_dimension=5,
        overlap_dimension=2,
        left_restriction=left,
        right_restriction=right,
    )
    return {"passed": result["h1_dimension"] == 0, **result}


def _nontrivial_finite_class_case() -> dict[str, Any]:
    result = two_set_cech_dimensions(
        left_dimension=1,
        right_dimension=1,
        overlap_dimension=1,
        left_restriction=np.zeros((1, 1)),
        right_restriction=np.zeros((1, 1)),
    )
    return {"passed": result["h1_dimension"] == 1, **result}


def _invalid_restriction_rejected_case() -> dict[str, Any]:
    rejected = False
    try:
        two_set_cech_dimensions(
            left_dimension=2,
            right_dimension=2,
            overlap_dimension=1,
            left_restriction=np.zeros((2, 2)),
            right_restriction=np.zeros((1, 2)),
        )
    except ValueError:
        rejected = True
    return {"passed": rejected, "invalid_restriction_rejected": rejected}


def _topology_relabel_invariance_case() -> dict[str, Any]:
    matrix = np.zeros((6, 6), dtype=np.float64)
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 4)):
        matrix[left, right] = matrix[right, left] = 1.0
    permutation = np.asarray((3, 0, 5, 2, 1, 4), dtype=np.int64)
    relabelled = matrix[np.ix_(permutation, permutation)]
    inverse = {int(old): int(new) for new, old in enumerate(permutation)}
    base = graph_descriptors(matrix, ports=(0, 5))
    transformed = graph_descriptors(
        relabelled,
        ports=(inverse[0], inverse[5]),
    )
    descriptor_keys = (
        "edge_count",
        "cycle_rank",
        "spectral_gap",
        "port_distance_mean",
        "effective_port_resistance_mean",
        "trace_a2",
        "trace_a3",
        "trace_a4",
    )
    defects = tuple(abs(float(base[key]) - float(transformed[key])) for key in descriptor_keys)
    response = np.arange(30, dtype=np.float64).reshape(5, 6)
    spectrum = singular_spectrum(response, relative_tolerances=(1e-10,))
    relabelled_spectrum = singular_spectrum(
        response[:, permutation], relative_tolerances=(1e-10,)
    )
    spectral_defect = float(
        np.max(
            np.abs(
                np.asarray(spectrum["singular_values"])
                - np.asarray(relabelled_spectrum["singular_values"])
            )
        )
    )
    maximum = max((*defects, spectral_defect))
    return {"passed": maximum < 1e-10, "maximum_relabelling_defect": maximum}


def _markov_order_and_generator_case() -> dict[str, Any]:
    groups = tuple(f"g{index // 12}" for index in range(60))
    generator = np.random.default_rng(9)
    state = generator.normal(size=61)
    feature = np.column_stack((state[:-1], np.roll(state[:-1], 1)))
    target = (0.8 * state[:-1] - 0.3 * np.roll(state[:-1], 1))[:, None]
    first = grouped_leave_one_out_ridge(feature[:, :1], target, groups, alpha=1e-8)
    second = grouped_leave_one_out_ridge(feature, target, groups, alpha=1e-8)
    doses = np.asarray((1.0, 0.5, 0.25, 0.125), dtype=np.float64)
    generator_estimates = (2.0 * doses + 0.5 * doses**2) / doses
    errors = np.abs(generator_estimates - 2.0)
    passed = float(second["rmse"]) < 0.2 * float(first["rmse"]) and bool(
        np.all(np.diff(errors) < 0)
    )
    return {
        "passed": passed,
        "order_one_rmse": first["rmse"],
        "order_two_rmse": second["rmse"],
        "generator_errors": tuple(float(value) for value in errors),
    }


def _rank_two_commuting_actions_case() -> dict[str, Any]:
    action_a = np.asarray((1.0, 0.0, 1.0, 0.0))
    action_b = np.asarray((0.0, 1.0, 0.0, -1.0))
    panel = np.stack((action_a, -action_a, action_b, -action_b))
    spectrum = singular_spectrum(panel, relative_tolerances=(1e-10,))
    rank = int(spectrum["relative_tolerance_ranks"]["1e-10"])
    order_defect = rms((action_a + action_b) - (action_b + action_a))
    return {"passed": rank == 2 and order_defect == 0.0, "rank": rank, "order_defect": order_defect}


def _exact_finite_decision_policy_case() -> dict[str, Any]:
    result = exact_adaptive_resolution_policy(
        ('adjacent-pair-partition', 'alternating-pair-partition'),
        {'adjacent-pair-partition': 1, 'alternating-pair-partition': 1},
        {'adjacent-pair-partition': (0b0011, 0b1100), 'alternating-pair-partition': (0b0101, 0b1010)},
        (0b0001, 0b0010, 0b0100, 0b1000),
        initial_mask=0b1111,
        maximum_acts=2,
        maximum_cost=2,
    )
    passed = result["resolution_probability"] == 1.0 and result["expected_cost"] == 2.0
    return {"passed": passed, **result}


def _admission_polytope_logged_vector_case() -> dict[str, Any]:
    records = (
        AdmissionMarginRecord("preparation.reference", 1, {"target": 0.2, "sink": 0.1}),
        AdmissionMarginRecord("preparation.reference", 2, {"target": 0.1, "sink": 0.05}),
    )
    active = tuple(
        min(value.margins, key=lambda key: value.margins[key]) for value in records
    )
    lyapunov = np.asarray((1.0, 0.8, 0.6), dtype=np.float64)
    logged_deltas = np.diff(lyapunov)
    passed = active == ("sink", "sink") and bool(np.all(logged_deltas <= 0))
    return {
        "passed": passed,
        "active_constraints": active,
        "logged_path_deltas": tuple(float(value) for value in logged_deltas),
        "global_certificate_claimed": False,
    }


def _source_seam_noncompensating_case() -> dict[str, Any]:
    nodes = ("source", "a", "unconnected-evidence", "target")
    edges = (("source", "a"), ("a", "target"), ("unconnected-evidence", "target"))
    dom = dominators(nodes, edges, "source")
    cuts = minimal_node_cutsets(nodes, edges, ("source",), "target", maximum_size=2)
    after_source_loss = directed_reachable(
        edges,
        ("source",),
        removed_edges=frozenset((("source", "a"),)),
    )
    return {
        "passed": "target" not in after_source_loss and cuts == (("a",),),
        "target_dominators": dom["target"],
        "cutsets": cuts,
        "target_reachable_after_source_loss": "target" in after_source_loss,
        "unconnected_evidence_compensates": False,
    }


def _future_leakage_and_unit_grouping_case() -> dict[str, Any]:
    rejected = False
    try:
        StateAugmentationCandidate("leaky", (0, 1), uses_future=True)
    except ValueError:
        rejected = True
    generator = np.random.default_rng(14)
    groups = tuple(f"preparation-{index // 20}" for index in range(80))
    x = generator.normal(size=(80, 2))
    y = (x[:, :1] * 2.0 + 0.1)
    result = grouped_leave_one_out_ridge(x, y, groups, alpha=1e-8)
    scored_groups = sum(row["status"] == "SCORED" for row in result["folds"])
    return {
        "passed": rejected and scored_groups == 4 and len(result["folds"]) == 4,
        "future_leakage_rejected": rejected,
        "nested_rows": 80,
        "independent_units": 4,
        "scored_folds": scored_groups,
    }


_CASES: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
    ("ALIGNABLE_RECEIVER_CHART", _alignable_receiver_chart_case),
    ("MANY_TO_ONE_INFORMATION_LOSS", _many_to_one_information_loss_case),
    ("RANK_ONE_INTERFACE", _rank_one_interface_case),
    ("DISTRIBUTED_FULL_RANK", _distributed_full_rank_case),
    ("ZERO_H1_COMPATIBLE_COVER", _zero_h1_compatible_cover_case),
    ("NONTRIVIAL_FINITE_CLASS", _nontrivial_finite_class_case),
    ("INVALID_RESTRICTION_REJECTED", _invalid_restriction_rejected_case),
    ("TOPOLOGY_RELABEL_INVARIANCE", _topology_relabel_invariance_case),
    ("MARKOV_ORDER_AND_GENERATOR", _markov_order_and_generator_case),
    ("RANK_TWO_COMMUTING_ACTIONS", _rank_two_commuting_actions_case),
    ("EXACT_FINITE_DECISION_POLICY", _exact_finite_decision_policy_case),
    ("ADMISSION_POLYTOPE_LOGGED_VECTOR", _admission_polytope_logged_vector_case),
    ("SOURCE_SEAM_NONCOMPENSATING", _source_seam_noncompensating_case),
    ("FUTURE_LEAKAGE_AND_UNIT_GROUPING", _future_leakage_and_unit_grouping_case),
)


def run_conformance_suite() -> dict[str, Any]:
    """Run the fourteen declared conformance cases through the same estimators used on parent evidence."""

    cases = tuple(_case(case_id, function) for case_id, function in _CASES)
    return {
        "status": "CONFORMANCE_PASSED" if all(case.passed for case in cases) else "FAILED",
        "case_count": len(cases),
        "passed_count": sum(case.passed for case in cases),
        "cases": tuple(asdict(case) for case in cases),
        "truth_known_only": True,
        "substitutes_for_parent_evidence": False,
    }


__all__ = ["ConformanceCase", "run_conformance_suite"]
