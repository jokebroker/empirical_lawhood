'Integrated terminal-record closeout for the integrated response closeout tranche.\n\nThis module combines already terminal child records.  It does not refit a\nchild, promote an outcome-visible parent, or pool numerical coefficients\nacross evidence worlds.\n'

from __future__ import annotations

from typing import Any, Mapping

from empirical_lawhood.kernel.provenance import ObjectIdentity

from .integrated_response_custody import (
    IntegratedCloseoutSourceExport,
    require_closeout_exports,
)


PRIMARY_TERMINAL_STATUSES = {
    'portfolio-entry-readiness': "READY_WITH_TYPED_NONBLOCKING_EXTERNAL_PREREQUISITES",
    'outcome-visible-response-grammar': "COMPLETE_NON_PROMOTABLE_OUTCOME_VISIBLE",
    'formal-structure-selection': "COMPLETE_NO_FORCED_WINNER",
    'truth-known-response-discrimination': "SUPPORTED",
    'independent-native-evaluation': "EVALUATION_GENERATED_SEALED",
    'topology-conditioned-generated-response': "SUPPORTED_GENERATED_TOPOLOGY_CONDITIONED_RESPONSE",
    'tokamak-signed-action-fibre': 'SIGNED_ACTION_FIBRE_RESOLVED',
    'battery-finite-memory-response': 'FINITE_CURVATURE_OR_MIXED_LOCAL_RESPONSE',
    'building-control-operand-readiness': 'PREREQUISITE_STOP_DELIVERY_AND_MODE_UNOBSERVABLE',
    'partial-response-transport': 'MIXED_TRANSPORT_WITH_INTERFACE_OBSTRUCTION',
    'truth-known-discovery-efficiency': 'DISCOVERY_EFFICIENCY_SUPPORTED_WITH_TYPED_NONIDENTIFIABILITY',
    'designed-substrate-task-and-structure': 'TASK_OR_STRUCTURE_ONLY_MIXED',
    'exact-simulator-prospective-control': 'EXACT_SIMULATOR_PROSPECTIVE_CONTROL_SUPPORTED',
    'physical-response-readiness': 'AUTHORITY_AND_APPARATUS_REQUIRED_PHYSICAL_NONATTEMPT',
    'synthetic-propulsion-interface-discovery': ('TRUTH_KNOWN_SHADOW_CORRECTED_ORACLE_SOURCE_AND_COMPUTABILITY_REQUIRED'),
}


def validate_terminal_records(records: Mapping[str, Mapping[str, Any]]) -> None:
    if set(records) != set(PRIMARY_TERMINAL_STATUSES):
        raise ValueError('integrated response closeout primary terminal inventory differs')
    for child_id, expected_status in PRIMARY_TERMINAL_STATUSES.items():
        record = records[child_id]
        observed = record.get("readiness_status") if child_id == 'portfolio-entry-readiness' else record.get("status")
        if observed != expected_status:
            raise ValueError(f"integrated response closeout terminal status differs for {child_id}: {observed}")


def _estimate(value: Mapping[str, Any]) -> float:
    return float(value["estimate"])


def build_integrated_closeout(
    *,
    terminals: Mapping[str, Mapping[str, Any]],
    details: Mapping[str, Mapping[str, Any]],
    retained_attempts: Mapping[str, Mapping[str, Any]],
    source_exports: tuple[IntegratedCloseoutSourceExport, ...],
    target_request: ObjectIdentity,
) -> dict[str, Any]:
    require_closeout_exports(
        terminals=terminals,
        details=details,
        retained_attempts=retained_attempts,
        source_exports=source_exports,
        target_request=target_request,
    )
    validate_terminal_records(terminals)
    required_details = {
        'outcome-visible-response-grammar',
        'formal-structure-selection',
        'truth-known-response-discrimination',
        'topology-conditioned-generated-response',
        'tokamak-signed-action-fibre',
        'battery-finite-memory-response',
        'partial-response-transport',
        'truth-known-discovery-efficiency',
        'designed-substrate-task-and-structure',
        'exact-simulator-prospective-control',
        'synthetic-propulsion-interface-discovery',
    }
    if set(details) != required_details:
        raise ValueError('integrated response closeout detailed terminal result inventory differs')
    retained_comparator = retained_attempts.get('synthetic-propulsion-comparator-ineligible')
    if not isinstance(retained_comparator, Mapping) or retained_comparator.get("status") != (
        'TRUTH_KNOWN_SHADOW_COMPARATOR_INELIGIBLE_SOURCE_AND_COMPUTABILITY_REQUIRED'
    ):
        raise ValueError('integrated response closeout did not retain the synthetic propulsion reference world attempt-1 terminal')
    if retained_comparator.get('retained_oracle_arm_claim_eligible') is True:
        raise ValueError('integrated response closeout retained comparator disposition differs')

    outcome_visible_response_grammar = details['outcome-visible-response-grammar']
    formal_structure_selection = details['formal-structure-selection']
    truth_known_response_discrimination = details['truth-known-response-discrimination']
    topology_conditioned_generated_response = details['topology-conditioned-generated-response']
    tokamak_signed_action_fibre = details['tokamak-signed-action-fibre']
    battery_finite_memory_response = details['battery-finite-memory-response']
    partial_response_transport = details['partial-response-transport']
    truth_known_discovery_efficiency = details['truth-known-discovery-efficiency']
    designed_substrate_task_and_structure = details['designed-substrate-task-and-structure']
    exact_simulator_prospective_control = details['exact-simulator-prospective-control']
    synthetic_propulsion_interface_discovery = details['synthetic-propulsion-interface-discovery']

    discovery_response = truth_known_discovery_efficiency["method_results"]["response_version_space"]
    designed_substrate_algebra = designed_substrate_task_and_structure["family_results"]["algebra_encoded"]
    conventional_reservoir = designed_substrate_task_and_structure["family_results"]["conventional_reservoir"]
    propulsion_response = synthetic_propulsion_interface_discovery["method_results"]["response_version_space"]
    propulsion_oracle = synthetic_propulsion_interface_discovery["method_results"]["oracle_upper_bound"]

    terminal_atlas = (
        {
            "cell_id": "atlas.outcome-visible-parent-grammar",
            "child_id": 'outcome-visible-response-grammar',
            "relation": "heterogeneous terminal contexts share typed experimental roles",
            "disposition": "SUPPORTED_NONPROMOTABLE",
            "evidence_world": "OUTCOME_VISIBLE_MIXED_PORTFOLIO",
            "ceiling": terminals['outcome-visible-response-grammar']["scientific_ceiling"],
        },
        {
            "cell_id": "atlas.truth-known-discrimination",
            "child_id": 'truth-known-response-discrimination',
            "relation": "formalization method separates planted positive and counterfeit cases",
            "disposition": "SUPPORTED",
            "evidence_world": "TRUTH_KNOWN",
            "ceiling": terminals['truth-known-response-discrimination']["scientific_ceiling"],
        },
        {
            "cell_id": "atlas.topology-conditioned-operator",
            "child_id": 'topology-conditioned-generated-response',
            "relation": "prepared topology and boundary condition select generated operator family",
            "disposition": "SUPPORTED_LOCAL",
            "evidence_world": "GENERATED_TRUTH_KNOWN",
            "ceiling": terminals['topology-conditioned-generated-response']["scientific_ceiling"],
        },
        {
            "cell_id": "atlas.gym-torax-signed-fibre",
            "child_id": 'tokamak-signed-action-fibre',
            "relation": "tested signed simulator action fibre exits rank zero",
            "disposition": "SUPPORTED_LOCAL",
            "evidence_world": "GYM_TORAX_SIMULATOR",
            "ceiling": terminals['tokamak-signed-action-fibre']["claim_ceiling"],
        },
        {
            "cell_id": "atlas.pybamm-finite-memory-curvature",
            "child_id": 'battery-finite-memory-response',
            "relation": "finite chart has memory and finite path defect without local-generator convergence",
            "disposition": "MIXED_LOCAL",
            "evidence_world": "PYBAMM_SIMULATOR",
            "ceiling": terminals['battery-finite-memory-response']["claim_ceiling"],
        },
        {
            "cell_id": "atlas.boptest-operand-stop",
            "child_id": 'building-control-operand-readiness',
            "relation": "delivered action and native realization mode are not observable",
            "disposition": "PREREQUISITE_STOP",
            "evidence_world": "BOPTEST_SOURCE_AUDIT",
            "ceiling": terminals['building-control-operand-readiness']["claim_ceiling"],
        },
        {
            "cell_id": "atlas.partial-transport",
            "child_id": 'partial-response-transport',
            "relation": "three nontrivial maps commute and one interface cut does not",
            "disposition": "MIXED_WITH_OBSTRUCTION",
            "evidence_world": "TRUTH_KNOWN_ANALYTIC",
            "ceiling": terminals['partial-response-transport']["claim_ceiling"],
        },
        {
            "cell_id": "atlas.discovery-efficiency",
            "child_id": 'truth-known-discovery-efficiency',
            "relation": "response-version selection reduces acts with typed nonidentifiability",
            "disposition": "SUPPORTED",
            "evidence_world": "TRUTH_KNOWN_GENERATED",
            "ceiling": terminals['truth-known-discovery-efficiency']["claim_ceiling"],
        },
        {
            "cell_id": "atlas.designed-substrate",
            "child_id": 'designed-substrate-task-and-structure',
            "relation": "algebra encoding improves tasks and recovery but fails complete signature transport",
            "disposition": "MIXED",
            "evidence_world": "TRUTH_KNOWN_GENERATED_SUBSTRATE",
            "ceiling": terminals['designed-substrate-task-and-structure']["claim_ceiling"],
        },
        {
            "cell_id": 'atlas.exact-simulator-prospective-control',
            "child_id": 'exact-simulator-prospective-control',
            "relation": "frozen constrained controller succeeds under exact generated simulator contract",
            "disposition": "SUPPORTED_EXACT_WORLD",
            "evidence_world": "GENERATED_SIMULATOR",
            "ceiling": terminals['exact-simulator-prospective-control']["claim_ceiling"],
        },
        {
            "cell_id": "atlas.physical-nonattempt",
            "child_id": 'physical-response-readiness',
            "relation": "physical transport and control lack apparatus and authority",
            "disposition": "AUTHORITY_AND_APPARATUS_REQUIRED",
            "evidence_world": "READINESS_AUDIT",
            "ceiling": terminals['physical-response-readiness']["claim_ceiling"],
        },
        {
            "cell_id": "atlas.icf-closure-shadow",
            "child_id": 'synthetic-propulsion-interface-discovery',
            "relation": "truth-known interface selection works while physical ICF seams lack sources",
            "disposition": "SHADOW_SUPPORTED_SOURCE_REQUIRED",
            "evidence_world": "TRUTH_KNOWN_ICF_LIKE_SHADOW",
            "ceiling": terminals['synthetic-propulsion-interface-discovery']["claim_ceiling"],
        },
    )

    non_entailments = (
        ("order", "action response", 'outcome-visible-response-grammar/truth-known-response-discrimination'),
        ("finite action response", "temporal composition", 'tokamak-signed-action-fibre/battery-finite-memory-response'),
        ("local composition", "context transport", 'partial-response-transport'),
        ("supported transport", "global gluing", 'partial-response-transport'),
        ("topology-conditioned generated law", "physical topology law", 'topology-conditioned-generated-response/physical-response-readiness'),
        ("held-out task performance", "complete structural signature", 'designed-substrate-task-and-structure'),
        ("fault recovery", "fault-source localization transport", 'designed-substrate-task-and-structure'),
        ("nominal receiver calibration", "cross-preparation receiver transport", 'designed-substrate-task-and-structure'),
        ('nonempty simulator receiver admission', 'physical receiver admission', 'exact-simulator-prospective-control/physical-response-readiness'),
        ('exact-simulator prospective control', "physical controller validation", 'exact-simulator-prospective-control/physical-response-readiness'),
        ("one-shot success", "repeated-cycle preservation", 'synthetic-propulsion-interface-discovery'),
        ("predictive memory", "finite path curvature", 'battery-finite-memory-response'),
        ("finite path curvature", "thermodynamic irreversibility", 'battery-finite-memory-response/physical-response-readiness'),
        ("response support", "receiver admission", 'outcome-visible-response-grammar/exact-simulator-prospective-control'),
        ("generated ICF-like selector success", "ICF or propulsion validation", 'synthetic-propulsion-interface-discovery'),
    )
    non_entailment_rows = tuple(
        {
            "source_claim": source,
            "nonentailed_claim": target,
            "witness_child_ids": witness,
        }
        for source, target, witness in non_entailments
    )

    formal_version_space = (
        {
            "candidate_id": 'unrelated-maps',
            "terminal_disposition": "SURVIVES_AS_LOWER_BOUND",
            "reason": "some contexts support only unrelated local maps",
        },
        {
            "candidate_id": 'partial-word-action',
            "terminal_disposition": "SUPPORTED_CONTEXT_LOCALLY_MIXED_GLOBALLY",
            "reason": "finite word sections recur but identity/composition are not universal",
        },
        {
            "candidate_id": 'temporal-cocycle',
            "terminal_disposition": "SUPPORTED_ON_SELECTED_LOCAL_SECTIONS_ONLY",
            "reason": "timing controls and some composition pass; no portfolio-wide cocycle",
        },
        {
            "candidate_id": 'context-fibred-system',
            "terminal_disposition": "BEST_INTERPRETIVE_ENVELOPE_NOT_FORMALLY_CLOSED",
            "reason": "typed contexts and maps exist, but cross-context arrows remain sparse",
        },
        {
            "candidate_id": 'overlap-gluing',
            "terminal_disposition": "MIXED_SUFFICIENT_BOUND_AND_INTERFACE_OBSTRUCTION",
            "reason": "one measured interface glues and one omitted interface fails globally",
        },
        {
            "candidate_id": 'topology-conditioned-mixture',
            "terminal_disposition": "SUPPORTED_GENERATED_FAMILY_ONLY",
            "reason": 'topology-conditioned mixtures recur in topology-conditioned generated response, not across evidence worlds',
        },
    )

    axiom_evidence = (
        ("typed_context", "DEFINITION_PLUS_CONFORMANCE", 'outcome-visible-response-grammar/truth-known-response-discrimination', "no universal coefficient"),
        ("finite_identity", "TRUTH_KNOWN_AND_LOCAL_SIMULATOR", 'topology-conditioned-generated-response/battery-finite-memory-response/partial-response-transport', "not all contexts"),
        ("partial_composition", "ANALYTIC_AND_LOCAL_SIMULATOR", 'battery-finite-memory-response/partial-response-transport', "support limited"),
        ("temporal_cocycle", "MIXED_LOCAL", 'truth-known-response-discrimination/tokamak-signed-action-fibre/battery-finite-memory-response', "no global stationarity"),
        (
            "context_naturality",
            "GENERATED_EXACT_RELABEL_AND_RECEIVER_MAP",
            'topology-conditioned-generated-response/partial-response-transport',
            "no physical transport",
        ),
        (
            "overlap_gluing",
            "ONE_SUFFICIENT_ANALYTIC_BOUND",
            'partial-response-transport',
            "one planted interface obstruction",
        ),
        ("topology_conditioning", "GENERATED_TRUTH_KNOWN", 'topology-conditioned-generated-response', "family and receiver qualified"),
        ("noncompensating_admission", "EXACT_SIMULATOR_CONFORMANCE", 'exact-simulator-prospective-control', "not physical"),
        ("mandatory_hold", "EXACT_SIMULATOR_CONFORMANCE", 'exact-simulator-prospective-control', "not physical"),
        ("physical_exchange_closure", "NO_SUPPORT", 'physical-response-readiness', "zero physical units/actions"),
        ("physical_context_transport", "NO_SUPPORT", 'physical-response-readiness', "apparatus and authority absent"),
        ("icf_to_propulsion_closure", "NO_SUPPORT", 'synthetic-propulsion-interface-discovery', "source and computability required"),
    )
    axiom_rows = tuple(
        {
            "axiom_id": axiom,
            "evidence_class": evidence,
            "witness_child_ids": witness,
            "ceiling_or_obstruction": obstruction,
        }
        for axiom, evidence, witness, obstruction in axiom_evidence
    )

    map_results = partial_response_transport["map_results"]
    supported_maps = tuple(
        {
            "map_id": row["map_id"],
            "functional_defect": row["functional_defect"],
            "disposition": row["disposition"],
        }
        for row in map_results
        if row["disposition"] == "EQUIVALENT"
    )
    obstructed_maps = tuple(
        {
            "map_id": row["map_id"],
            "functional_defect": row["functional_defect"],
            "disposition": row["disposition"],
        }
        for row in map_results
        if row["disposition"] != "EQUIVALENT"
    )

    topology_exchanges = topology_conditioned_generated_response["topology_exchanges"]
    receiver_hierarchy = topology_conditioned_generated_response["receiver_hierarchy"]
    topology_read = {
        "topology_conditioned": {
            "tree_vs_cycle_material_all_four_normalizations_and_laws": all(
                row["material"] for row in topology_exchanges["tree-vs-cycle"]
            ),
            "open_vs_periodic_material_all_four_normalizations_and_laws": all(
                row["material"] for row in topology_exchanges["boundary-open-vs-periodic-grid"]
            ),
            "operator_improvement_all_blocks": topology_conditioned_generated_response[
                "topology_conditioned_operator_improvement_all_blocks"
            ],
            "evidence_world": "GENERATED_TRUTH_KNOWN",
        },
        "geometry_mediated_or_null": {
            "degree_matched_cube_vs_mobius_material": any(
                row["material"] for row in topology_exchanges["degree-matched-cube-vs-mobius"]
            ),
            "path_vs_cycle_material": any(
                row["material"] for row in topology_exchanges["path-vs-cycle"]
            ),
            "modular_vs_cycle_material": any(
                row["material"] for row in topology_exchanges["modular-vs-cycle"]
            ),
        },
        "receiver_manufactured": {
            "classification_accuracy_by_receiver": tuple(
                {
                    "receiver": row["receiver"],
                    "accuracy": row["topology_classification_accuracy"],
                }
                for row in receiver_hierarchy
            ),
            'designed_substrate_task_and_structure_algebra_receiver_calibration_coverage': designed_substrate_algebra["metrics"][
                "receiver_calibration_coverage"
            ],
            'designed_substrate_task_and_structure_fault_localization_accuracy': designed_substrate_algebra["metrics"][
                "fault_localization_accuracy"
            ],
        },
    }

    evidence_thresholds = {
        'truth_known_response_discrimination_independent_unit_curve': truth_known_response_discrimination["evidence_prefix_curves"],
        'truth_known_response_discrimination_transformation_coverage_curve': truth_known_response_discrimination["transformation_coverage_curve"],
        "universal_sample_threshold_claimed": False,
        "interpretation": (
            "stability was nonmonotone at prefixes 2/3/4/6/8 and formalization "
            "completed only when all seven planted transformation axes were tested"
        ),
    }

    separated_outcomes = {
        "discovery_efficiency": {
            "grade": truth_known_discovery_efficiency["discovery_efficiency_grade"],
            "independent_world_count": truth_known_discovery_efficiency["independent_evaluation_world_count"],
            "response_acts_used": discovery_response["metrics"]["acts_used"],
            "false_promotion": discovery_response["metrics"]["false_promotion"],
        },
        "designed_substrate_task_performance": {
            "algebra_task_score": designed_substrate_algebra["metrics"]["aggregate_task_score"],
            "conventional_task_score": conventional_reservoir["metrics"]["aggregate_task_score"],
            "heldout_task_value_supported": designed_substrate_task_and_structure["heldout_task_value_supported"],
        },
        "designed_substrate_structural_realization": {
            "complete_signature_rate": designed_substrate_algebra["metrics"]["signature_realized"],
            "fault_localization_accuracy": designed_substrate_algebra["metrics"]["fault_localization_accuracy"],
            "signature_realization_supported": designed_substrate_task_and_structure["signature_realization_supported"],
        },
        'prospective_control': {
            'prospective_control_supported': exact_simulator_prospective_control['prospective_control_supported'],
            "target_success_rate": exact_simulator_prospective_control["metrics"]["supported_target_success_rate"],
            "hold_calibration": exact_simulator_prospective_control["metrics"]["unsupported_hold_calibration"],
            "constraint_violation_count": exact_simulator_prospective_control["constraint_violation_count"],
            "evidence_world": exact_simulator_prospective_control["claim_ceiling"],
        },
        "icf_like_discovery": {
            "observable_class_resolution": propulsion_response["equivalence_resolution_rate"],
            "oracle_cost": propulsion_oracle["mean_cost"],
            "response_cost": propulsion_response["mean_cost"],
            "true_nonidentifiable_preparations": synthetic_propulsion_interface_discovery["true_nonidentifiable_preparation_count"],
            "physical_icf_validated": synthetic_propulsion_interface_discovery["icf_physics_validated"],
        },
    }

    maximum_claim = (
        "In evaluator-known analytic and generated open-system families and two "
        "fresh simulator-local charts, finite response evidence is best represented "
        "by context-indexed partial sections with explicitly tested identities, "
        "compositions, topology conditions, receiver maps and gluing obstructions. "
        "This typed structure improved scientific experiment selection in two "
        "truth-known benchmarks and enabled one prospectively validated controller "
        "inside its exact generated simulator contract. Intentional algebra encoding "
        "improved held-out computation and recovery but did not realize its complete "
        "transport signature. No physical transport, thermodynamic closure, universal "
        "response algebra, ICF physics, propulsion system or mission reliability was "
        "established."
    )

    return {
        "status": 'INTEGRATED_RESPONSE_CLOSEOUT_COMPLETE',
        "primary_terminal_child_count": len(terminals),
        "retained_superseded_attempt_count": len(retained_attempts),
        "all_primary_children_terminal": True,
        "cross_substrate_numeric_pooling": False,
        "smallest_supported_formal_object": (
            "EVIDENCE_ENRICHED_CONTEXT_INDEXED_PARTIAL_MAP_DIAGRAM_OF_FINITE_RESPONSE_SECTIONS"
        ),
        "category_claim_supported": False,
        "functor_claim_supported": False,
        "sheaf_claim_supported": False,
        "universal_response_algebra_supported": False,
        "terminal_relational_atlas": terminal_atlas,
        "non_entailment_graph": non_entailment_rows,
        "formal_version_space": formal_version_space,
        "forced_formal_winner": False,
        "axiom_evidence": axiom_rows,
        "supported_context_maps": supported_maps,
        "obstructed_context_maps": obstructed_maps,
        "gluing_results": partial_response_transport["local_section_overlap_records"],
        "evidence_thresholds": evidence_thresholds,
        "topology_geometry_receiver_separation": topology_read,
        "separated_scientific_outcomes": separated_outcomes,
        "attempt_integrity": {
            'retained_propulsion_oracle_arm_claim_eligible': False,
            'propulsion_corrected_oracle_contract_passed': synthetic_propulsion_interface_discovery[
                "corrected_oracle_contract_passed"
            ],
        },
        "physical_evidence_present": False,
        "physical_action_count": terminals['physical-response-readiness']["physical_action_count"],
        'exact_simulator_prospective_control_present': exact_simulator_prospective_control['prospective_control_supported'],
        "icf_physics_validated": synthetic_propulsion_interface_discovery["icf_physics_validated"],
        "propulsion_system_validated": synthetic_propulsion_interface_discovery["propulsion_system_validated"],
        "maximum_claim": maximum_claim,
        "claim_ceiling": "MIXED_ANALYTIC_GENERATED_AND_SIMULATOR_NO_PHYSICAL_PROMOTION",
        "scientific_readout": {
            "atlas_cell_count": len(terminal_atlas),
            "non_entailment_count": len(non_entailment_rows),
            'outcome_visible_response_grammar_parent_atlas_cell_count': outcome_visible_response_grammar["atlas_cell_count"],
            'outcome_visible_response_grammar_missing_arrow_count': outcome_visible_response_grammar["missing_arrow_count"],
            'outcome_visible_response_grammar_witnessed_non_entailment_count': outcome_visible_response_grammar["witnessed_non_entailment_count"],
            'formal_structure_selection_all_candidate_ids_survive': len(formal_structure_selection["surviving_candidate_ids"]) == 6,
            'tokamak_signed_action_fibre_rank_zero_exited': tokamak_signed_action_fibre["action_fibre_exits_rank_zero"],
            'battery_finite_memory_response_generator_converged': battery_finite_memory_response[
                "convergent_local_generator_candidate_all_operational"
            ],
            'partial_response_transport_supported_nontrivial_map_count': partial_response_transport["supported_nontrivial_transport_count"],
            'partial_response_transport_failed_nontrivial_map_count': partial_response_transport["failed_nontrivial_transport_count"],
            'designed_substrate_task_and_structure_complete_signature_supported': designed_substrate_task_and_structure["signature_realization_supported"],
            'exact_simulator_prospective_control_target_success': _estimate(exact_simulator_prospective_control["metrics"]["supported_target_success_rate"]),
            'synthetic_propulsion_interface_discovery_repeated_cycle_preservation': _estimate(
                synthetic_propulsion_interface_discovery["reliability"]["final_cycle_preservation_rate"]
            ),
        },
    }
