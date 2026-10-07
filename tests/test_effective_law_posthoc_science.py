from __future__ import annotations

from empirical_lawhood.planning.posthoc import POSTHOC_ANALYSIS_IDS

from empirical_lawhood.adapters.methods.effective_law_comparison.analysis import execute_analysis, execute_integrated_skeptic, execute_skeptic, synthesize


def _document(schema: str, value: dict[str, object]) -> dict[str, object]:
    return {"schema": schema, "version": "1.0.0", "value": value}


def _panel(target: str) -> dict[str, object]:
    target_gate = "target-temperature-band" if target == "cantera" else "downstream-target-band"
    sink_gates = (
        ("co-ceiling", "element-closure", "minimum-conversion", "peak-temperature-ceiling")
        if target == "cantera"
        else (
            "balance-error-ceiling",
            "boundary-flux-ceiling",
            "nonnegativity-tolerance",
            "peak-field-ceiling",
        )
    )
    margins = [
        {
            "schema": 'empirical-lawhood/kernel/named-decimal',
            "version": "1.0.0",
            "value": {
                "unit": "native-margin",
                "value": {"decimal": "2"},
                "value_id": target_gate,
            },
        }
    ]
    margins.extend(
        {
            "schema": 'empirical-lawhood/kernel/named-decimal',
            "version": "1.0.0",
            "value": {
                "unit": "native-margin",
                "value": {"decimal": "-1" if index == 0 else "1"},
                "value_id": gate,
            },
        }
        for index, gate in enumerate(sink_gates)
    )
    return _document(
        'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-target-panel',
        {
            "complete_units": [
                {
                    "schema": 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-complete-unit-result',
                    "version": "1.0.0",
                    "value": {
                        "complete_unit_id": f"{target}.unit-001",
                        "context_decisions": [
                            {
                                "schema": 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-context-decision',
                                "version": "1.0.0",
                                "value": {
                                    "decision_id": "decision.test",
                                    "denominator_id": "denominator.test",
                                    "history_id": "history.test",
                                    "horizon_id": "horizon.test",
                                    "hold_action_id": "action.hold",
                                    "hold_viable": False,
                                },
                            }
                        ],
                        "conditions": [
                            {
                                "schema": 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-condition-result',
                                "version": "1.0.0",
                                "value": {
                                    "condition_id": "condition.test",
                                    "action_id": "action.active",
                                    "action": {
                                        "schema": 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-action-realization',
                                        "version": "1.0.0",
                                        "value": {"action_id": "action.active"},
                                    },
                                    "denominator_id": "denominator.test",
                                    "history_id": "history.test",
                                    "horizon_id": "horizon.test",
                                    "fibre_admitted": False,
                                    "gate_margins": margins,
                                    "stopped": False,
                                    "support_state": "inside-support",
                                },
                            }
                        ],
                    },
                }
            ]
        },
    )


def test_gate_semantics_counterexample_precedence_for_compensation_and_gate_deletion() -> None:
    documents = {
        "selective-dependence-response-cantera-panel": _panel("cantera"),
        "selective-dependence-response-fipy-panel": _panel("fipy"),
    }
    result = execute_analysis(
        "analysis.noncompensating-gate-semantics",
        documents,
        ("selective-dependence-response-cantera", "selective-dependence-response-fipy"),
    )
    value = result["value"]
    assert value["status"] == "OPPOSED"
    counterexample_ids = {row["counterexample_id"] for row in value["counterexamples"]}
    assert any("equal-weight-sign-compensation" in value for value in counterexample_ids)
    assert any(
        f"leave-out-{('co-ceiling' if 'cantera' in value else 'balance-error-ceiling')}" in value
        for value in counterexample_ids
    )
    assert value["no_cross_partition_numerical_pooling"] is True


def test_predictive_fit_does_not_compensate_semantic_role_loss() -> None:
    denominator = _document(
        'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-denominator-selection',
        {
            "assessments": [
                {
                    "schema": 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-denominator-candidate-assessment',
                    "version": "1.0.0",
                    "value": {
                        "adequate": False,
                        "alternative": "MERGED",
                        "calibration_accuracy": {"decimal": "0.99"},
                        "failure_codes": ["semantic-preservation-failed"],
                        "omitted_role_id": "D",
                        "semantic_preservation": False,
                    },
                }
            ]
        },
    )
    documents = {
        "selective-dependence-response-cantera-denominator": denominator,
        "selective-dependence-response-fipy-denominator": denominator,
        "battery-reduced-observation-fiber": _document(
            "empirical-lawhood/methods/effective-law-comparison/finite-fiber-science-fixture",
            {"rows": [{"cell": "CLOSE_DIVERGED"}, {"cell": "CLOSE_CLOSED"}]},
        ),
    }

    result = execute_analysis(
        "analysis.semantic-role-preservation",
        documents,
        ("selective-dependence-response-cantera", "selective-dependence-response-fipy", "battery-reduced-observation"),
    )
    value = result["value"]
    assert value["status"] == "OPPOSED"
    assert len(value["counterexamples"]) == 2
    assert value["estimands"]["battery_reduced_observation_fiber_cells"] == {
        "CLOSE_CLOSED": 1,
        "CLOSE_DIVERGED": 1,
    }


def test_synthesis_preserves_negative_and_nonpromotable_statuses() -> None:
    results = []
    reports = []
    for analysis_id in POSTHOC_ANALYSIS_IDS:
        result = _document(
            'empirical-lawhood/runtime/analysis-result',
            {
                "analysis_id": analysis_id,
                "claim_promotion_allowed": False,
                "complete_family_executed": True,
                "counterexample_precedence": True,
                "counterexamples": [{"counterexample_id": f"counterexample-{analysis_id}"}],
                "estimands": {"comparator": "present", "denominator": "typed", "power": "bounded"},
                "interpretation": "Negative outcomes remain evidence.",
                "no_cross_partition_numerical_pooling": True,
                "outcome_access": "evaluation-revealed",
                "parent_ids": ["parent.test"],
                "status": "OPPOSED",
                "typed_stops": [{"reason_code": "UNEVALUABLE_UNIT_STRUCTURE"}],
                "visibility_ceiling": "OUTCOME_VISIBLE",
            },
        )
        results.append(result)
        reports.append(execute_skeptic(result))
    integrated = execute_integrated_skeptic(reports)
    synthesis = synthesize(results, integrated)

    assert synthesis["value"]["claim_ceiling"].startswith("NON_PROMOTABLE")
    assert synthesis["value"]["no_cross_partition_numerical_pooling"] is True
    assert any(
        row["hypothesis_id"] == "predictive-collapse-to-control" and row["disposition"] == "OPPOSED"
        for row in synthesis["value"]["hypotheses"]
    )
