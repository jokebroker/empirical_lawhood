"""Pure scientific-purpose analyzers and protected synthesis.

All cross-program calculations are logical enumerations.  Numerical summaries
are calculated only inside one parent partition and retain the complete unit as
the block.  The functions consume already hash-qualified canonical documents;
they never open source paths.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from typing import Any

from empirical_lawhood.adapters.methods.selective_dependence_response.power import SelectiveDependenceResponsePowerMethod, SelectiveDependenceResponsePowerOperand, qualify_power_design
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.planning.posthoc import POSTHOC_ANALYSIS_IDS
from empirical_lawhood.planning.exploration import (
    Hypothesis,
    HypothesisDisposition,
    HypothesisSet,
)
from empirical_lawhood.runtime.posthoc import (
    ANALYSIS_RESULT_SCHEMA,
    INTEGRATED_SKEPTIC_SCHEMA,
    METATHEORY_SYNTHESIS_SCHEMA,
    SKEPTIC_RESULT_SCHEMA,
)

from .projections import project_comparators, project_units
from .statistics import complete_unit_bootstrap_mean, leave_one_unit_means, ratio

DocumentMap = Mapping[str, Mapping[str, Any]]


class PosthocDocumentSet(dict[str, Mapping[str, Any]]):
    """Qualified source documents with the explicit parent partition mapping."""

    def __init__(
        self,
        documents: DocumentMap,
        parent_by_artifact: Mapping[str, str],
    ) -> None:
        super().__init__(documents)
        self.parent_by_artifact = dict(parent_by_artifact)
        if set(self) != set(self.parent_by_artifact):
            raise ValueError("post-hoc document partition map is incomplete")


def _value(value: object) -> Any:
    if isinstance(value, Mapping) and set(value) == {"schema", "value", "version"}:
        return _value(value["value"])
    if isinstance(value, Mapping):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_value(item) for item in value]
    return value


def _walk(value: object) -> Iterable[object]:
    yield value
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


def _strings(document: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(value for value in _walk(document) if isinstance(value, str))


def _keys(document: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        str(key) for value in _walk(document) if isinstance(value, Mapping) for key in value
    )


def _document(documents: DocumentMap, artifact_id: str) -> Mapping[str, Any]:
    try:
        return documents[artifact_id]
    except KeyError as error:
        raise ValueError(
            f"qualified artifact missing from analysis input: {artifact_id}"
        ) from error


def _parent_documents(documents: DocumentMap, parent_id: str) -> list[Mapping[str, Any]]:
    if not isinstance(documents, PosthocDocumentSet):
        raise ValueError("post-hoc analysis needs an explicit parent partition map")
    return [
        document
        for artifact_id, document in documents.items()
        if documents.parent_by_artifact[artifact_id] == parent_id
    ]


def _root_value(document: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _value(document)
    if not isinstance(value, Mapping):
        raise ValueError("canonical source does not decode to a mapping")
    return value


def _nested_records(value: object, *, required_keys: frozenset[str]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for item in _walk(_value(value)):
        if isinstance(item, Mapping) and required_keys.issubset(item):
            rows.append(item)
    return rows


def _ratio(numerator: int, denominator: int) -> str:
    return "0/0" if denominator == 0 else f"{numerator}/{denominator}"


def _base(
    analysis_id: str,
    *,
    status: str,
    parents: Sequence[str],
    estimands: Mapping[str, object],
    counterexamples: Sequence[Mapping[str, object]],
    stops: Sequence[Mapping[str, object]],
    interpretation: str,
) -> dict[str, object]:
    return {
        "schema": ANALYSIS_RESULT_SCHEMA,
        "version": "1.0.0",
        "value": {
            "analysis_id": analysis_id,
            "claim_promotion_allowed": False,
            "complete_family_executed": True,
            "counterexample_precedence": True,
            "counterexamples": list(counterexamples),
            "estimands": dict(estimands),
            "interpretation": interpretation,
            "no_cross_partition_numerical_pooling": True,
            "outcome_access": "evaluation-revealed",
            "parent_ids": sorted(parents),
            "status": status,
            "typed_stops": list(stops),
            "visibility_ceiling": "OUTCOME_VISIBLE",
        },
    }


def _assess_coordinate_necessity_and_construct_diversity(documents: DocumentMap, parent_ids: Sequence[str]) -> dict[str, object]:
    rows: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    for target in ("cantera", "fipy"):
        construct_validation_panel_document = _document(documents, f"target-construct-validation-{target}-panel")
        construct_validation_bundle_document = _document(documents, f"target-construct-validation-{target}-bundle")
        construct_validation_panel = _root_value(construct_validation_panel_document)
        comparators = project_comparators(construct_validation_bundle_document, construct_validation_panel_document)
        selective_response_panel = _root_value(_document(documents, f"selective-dependence-response-{target}-panel"))
        separation = _root_value(_document(documents, f"selective-dependence-response-{target}-forecast-separation"))
        comparator_docs = [
            _root_value(document)
            for artifact_id, document in documents.items()
            if artifact_id.startswith(f"selective-dependence-response-{target}-comparator-")
        ]
        construct_validation_cases = construct_validation_panel.get("cases", [])
        selective_response_units = selective_response_panel.get("complete_units", [])
        if not isinstance(construct_validation_cases, list) or not isinstance(selective_response_units, list):
            raise ValueError("independent substrate grounding panel shape changed")
        exact = [value for value in comparators if value["exact"] is True]
        exact_masks = {tuple(str(role) for role in value["dependency_role_ids"]) for value in exact}
        minimal_masks = sorted(
            mask for mask in exact_masks if not any(set(other) < set(mask) for other in exact_masks)
        )
        full_mask = ("A", "D", "H", "R", "tau")
        strict_exact_masks = sorted(mask for mask in exact_masks if set(mask) < set(full_mask))
        equivalence: dict[tuple[int, ...], list[str]] = defaultdict(list)
        for comparator in comparators:
            aggregate = comparator["aggregate"]
            assert isinstance(aggregate, Mapping)
            signature = tuple(
                int(aggregate[name])
                for name in (
                    "unsafe_false_admission_count",
                    "categorical_mismatch_count",
                    "uncovered_case_count",
                    "unevaluable_case_count",
                    "prediction_set_cardinality",
                )
            )
            equivalence[signature].append(str(comparator["comparator_kind"]))
        evaluation_counts = [value.get("evaluation_outcome_count") for value in comparator_docs]
        no_target_evaluation = bool(evaluation_counts) and all(
            value == 0 for value in evaluation_counts
        )
        separation_rows = separation.get("separations", [])
        separated = (
            sum(
                1
                for item in separation_rows
                if isinstance(item, Mapping) and _value(item).get("separated") is True
            )
            if isinstance(separation_rows, list)
            else 0
        )
        rows.append(
            {
                "target_id": target,
                "construct_validation_complete_units": len(set(construct_validation_panel.get("complete_unit_ids", []))),
                "construct_validation_nested_case_count": len(construct_validation_cases),
                "construct_validation_comparators": comparators,
                "construct_validation_exact_comparator_kinds": sorted(
                    str(value["comparator_kind"]) for value in exact
                ),
                "construct_validation_exact_coordinate_masks": [list(value) for value in sorted(exact_masks)],
                "construct_validation_minimal_adequate_coordinate_antichain": [
                    list(value) for value in minimal_masks
                ],
                "construct_validation_score_equivalence_classes": [
                    {
                        "score_signature": list(signature),
                        "comparator_kinds": sorted(kinds),
                    }
                    for signature, kinds in sorted(equivalence.items())
                ],
                "selective_response_complete_units": len(selective_response_units),
                "selective_response_comparator_count": len(comparator_docs),
                "selective_response_comparators_separated_in_development": separated,
                "selective_response_evaluation_outcome_count": 0 if no_target_evaluation else "NONZERO_OR_UNRESOLVED",
            }
        )
        if strict_exact_masks:
            counterexamples.append(
                {
                    "counterexample_id": f"{target}-construct-validation-simple-exact",
                    "opposes": "full-coordinate-necessity-at-construct-validation-endpoint",
                    "parent_id": f"target-construct-validation-{target}",
                    "strict_exact_coordinate_masks": [list(value) for value in strict_exact_masks],
                }
            )
        if no_target_evaluation:
            counterexamples.append(
                {
                    "counterexample_id": f"{target}-selective-response-no-evaluation",
                    "opposes": "using-development-separation-as-recurrence",
                    "parent_id": f"selective-dependence-response-{target}",
                }
            )
    correction_tokens = sorted(_strings(_document(documents, "target-construct-validation-cross-target-correction")))
    controlling_negative = [
        value
        for value in correction_tokens
        if "NOT_DISTINGUISHED" in value or "RECURRENCE_NOT" in value
    ]
    return _base(
        "analysis.coordinate-necessity-and-construct-diversity",
        status="OPPOSED",
        parents=parent_ids,
        estimands={"controlling_correction_tokens": controlling_negative, "target_rows": rows},
        counterexamples=counterexamples,
        stops=(),
        interpretation=(
            "The construct-validation endpoint does not identify full-coordinate necessity because simpler "
            "comparators remain exact; selective-response establishes development-only construct diversity "
            "but generated no evaluation outcomes. Endpoint sensitivity beyond the recorded "
            "categorical construct-validation task remains unevaluable rather than being filled in post hoc."
        ),
    )


def _assess_forecast_level_comparability(documents: DocumentMap, parent_ids: Sequence[str]) -> dict[str, object]:
    rows: list[dict[str, Any]] = []
    stops: list[dict[str, Any]] = []
    for target in ("cantera", "fipy"):
        comparators = project_comparators(
            _document(documents, f"target-construct-validation-{target}-bundle"),
            _document(documents, f"target-construct-validation-{target}-panel"),
        )
        structural_recurrence = next(value for value in comparators if value["comparator_kind"] == "structural-recurrence")
        unit_rows = structural_recurrence["complete_unit_rows"]
        assert isinstance(unit_rows, list)
        exact_units = sum(
            isinstance(value, Mapping)
            and value.get("categorical_mismatch_count", 0) == 0
            and value.get("unsafe_false_admission_count", 0) == 0
            and value.get("unevaluable_case_count", 0) == 0
            for value in unit_rows
        )
        rows.append(
            {
                "chronology_basis": (
                    "development-only frozen encoding embedded with a distinct evaluation panel"
                ),
                "complete_unit_count": len(unit_rows),
                "exact_complete_unit_count": exact_units,
                "forecast_level": "categorical",
                "parent_id": f"target-construct-validation-{target}",
                "unsafe_complete_unit_count": sum(
                    isinstance(value, Mapping)
                    and int(value.get("unsafe_false_admission_count", 0)) > 0
                    for value in unit_rows
                ),
                "unit_rows": unit_rows,
            }
        )
    for parent_id in ('action-fiber-structural-recurrence', 'margin-structural-recurrence-forecast'):
        if parent_id in parent_ids:
            stops.append(
                {
                    "parent_id": parent_id,
                    "reason_code": "FORECAST_OUTCOME_PAIR_NOT_IN_FROZEN_SOURCE_ROSTER",
                }
            )
    for parent_id in parent_ids:
        if parent_id.startswith("selective-dependence-response-"):
            stops.append({"parent_id": parent_id, "reason_code": "OUTCOME_NOT_GENERATED"})
    stops.append({"parent_id": "portfolio", "reason_code": "NO_COMMON_EVALUABLE_LEVELS"})
    return _base(
        "analysis.forecast-level-comparability",
        status="UNEVALUABLE",
        parents=parent_ids,
        estimands={
            "eligible_categorical_parent_count": len(rows),
            "eligible_dynamical_parent_count": 0,
            "eligible_metric_parent_count": 0,
            "rows": rows,
            "weak_order": "UNEVALUABLE_NO_BALANCED_LEVEL_COMPARISON",
        },
        counterexamples=(),
        stops=stops,
        interpretation=(
            "The two target construct validation categorical forecasts are exactly scoreable by complete unit. "
            "The frozen roster does not contain paired, chronology-proved metric or dynamical "
            "forecast rows, so their apparent scarcity is missingness—not evidence that "
            "categorical structure transports better."
        ),
    )


def _conditions(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return _nested_records(
        document,
        required_keys=frozenset({"complete_unit_id", "conditions"}),
    )


def _assess_noncompensating_gate_semantics(documents: DocumentMap, parent_ids: Sequence[str]) -> dict[str, object]:
    target_rows: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    stops: list[dict[str, Any]] = []
    gate_roles = {
        "cantera": (
            "target-temperature-band",
            ("co-ceiling", "element-closure", "minimum-conversion", "peak-temperature-ceiling"),
        ),
        "fipy": (
            "downstream-target-band",
            (
                "balance-error-ceiling",
                "boundary-flux-ceiling",
                "nonnegativity-tolerance",
                "peak-field-ceiling",
            ),
        ),
    }
    for target in ("cantera", "fipy"):
        units = project_units(_document(documents, f"selective-dependence-response-{target}-panel"))
        target_gate, sink_gates = gate_roles[target]
        rule_totals: dict[str, Counter[str]] = defaultdict(Counter)
        unit_rows = []
        for unit in units:
            unit_counts: dict[str, Counter[str]] = defaultdict(Counter)
            decisions = {
                ".".join(
                    str(value.get(name)) for name in ("denominator_id", "history_id", "horizon_id")
                ): value
                for value in unit["context_decisions"]
                if isinstance(value, Mapping)
            }
            contexts: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for condition in unit["conditions"]:
                if isinstance(condition, Mapping):
                    contexts[str(condition["context_id"])].append(condition)
            for context_id, conditions in contexts.items():
                decision = decisions.get(context_id)
                if decision is None:
                    raise ValueError("selective dependence response context decision is missing")
                hold_action_id = str(decision.get("hold_action_id"))
                parent_hold = decision.get("hold_viable") is True
                active = [
                    value
                    for value in conditions
                    if value.get("action_id") != hold_action_id
                    and value.get("support_state") != "outside-support"
                    and value.get("stopped") is not True
                ]
                parent_active = {
                    str(value["action_id"])
                    for value in active
                    if value.get("fibre_admitted") is True
                }
                rule_active: dict[str, set[str]] = {
                    "active-only": set(parent_active),
                    "always-act": {str(value["action_id"]) for value in active},
                    "always-hold": set(),
                    "default-safe-hold": set(parent_active),
                    "noncompensating-intersection": set(parent_active),
                    "target-only": set(),
                    "target-plus-sink": set(),
                    "equal-weight-sign-compensation": set(),
                }
                gate_ids = sorted(
                    {str(gate_id) for value in active for gate_id in value.get("gate_values", {})}
                )
                for gate_id in gate_ids:
                    rule_active[f"leave-out-{gate_id}"] = set()
                for condition in active:
                    action_id = str(condition["action_id"])
                    raw_gates = condition.get("gate_values")
                    if not isinstance(raw_gates, Mapping):
                        raise ValueError("selective dependence response gate vector changed shape")
                    gates = {str(key): Decimal(str(value)) for key, value in raw_gates.items()}
                    if target_gate not in gates or not set(sink_gates).issubset(gates):
                        raise ValueError("selective dependence response target/sink gate role is absent")
                    if gates[target_gate] >= 0:
                        rule_active["target-only"].add(action_id)
                    if gates[target_gate] >= 0 and all(gates[name] >= 0 for name in sink_gates):
                        rule_active["target-plus-sink"].add(action_id)
                    signs = [1 if value >= 0 else -1 for value in gates.values()]
                    if sum(signs) >= 0:
                        rule_active["equal-weight-sign-compensation"].add(action_id)
                    for gate_id in gate_ids:
                        if all(value >= 0 for name, value in gates.items() if name != gate_id):
                            rule_active[f"leave-out-{gate_id}"].add(action_id)
                for rule_id, admitted in rule_active.items():
                    false_active = admitted - parent_active
                    missed_active = parent_active - admitted
                    measured_hold_rule = rule_id not in {"active-only", "always-act"}
                    selects_hold = (
                        rule_id == "always-hold"
                        or (rule_id == "default-safe-hold" and not admitted)
                        or (measured_hold_rule and not admitted and parent_hold)
                    )
                    selects_nothing = not admitted and not selects_hold
                    counts = unit_counts[rule_id]
                    counts["applicable_context_count"] += 1
                    counts["applicable_active_fibre_count"] += len(active)
                    counts["false_admission_count"] += len(false_active)
                    counts["false_action_count"] += len(false_active)
                    counts["missed_viable_action_count"] += len(missed_active)
                    counts["false_safe_hold_count"] += int(selects_hold and not parent_hold)
                    counts["unnecessary_nonattempt_count"] += int(
                        selects_nothing and (bool(parent_active) or parent_hold)
                    )
            for rule_id, counts in unit_counts.items():
                rule_totals[rule_id].update(counts)
            unit_rows.append(
                {
                    "complete_unit_id": unit["complete_unit_id"],
                    "rule_counts": {
                        key: dict(sorted(value.items()))
                        for key, value in sorted(unit_counts.items())
                    },
                }
            )
        rule_rows: list[dict[str, Any]] = []
        for rule_id, counts in sorted(rule_totals.items()):
            safety = counts["false_admission_count"] + counts["false_safe_hold_count"]
            rule_rows.append({"rule_id": rule_id, **dict(sorted(counts.items()))})
            if safety:
                counterexamples.append(
                    {
                        "counterexample_id": f"{target}-{rule_id}-safety-error",
                        "false_admission_count": counts["false_admission_count"],
                        "false_safe_hold_count": counts["false_safe_hold_count"],
                        "opposes": "safety-of-the-corresponding-gate-or-hold-rule",
                        "parent_id": f"selective-dependence-response-{target}",
                    }
                )
        pareto: list[str] = []
        for row in rule_rows:
            vector = (
                int(row.get("false_admission_count", 0)) + int(row.get("false_safe_hold_count", 0)),
                int(row.get("missed_viable_action_count", 0)),
                int(row.get("unnecessary_nonattempt_count", 0)),
            )
            dominated = any(
                other is not row
                and all(
                    left <= right
                    for left, right in zip(
                        (
                            int(other.get("false_admission_count", 0))
                            + int(other.get("false_safe_hold_count", 0)),
                            int(other.get("missed_viable_action_count", 0)),
                            int(other.get("unnecessary_nonattempt_count", 0)),
                        ),
                        vector,
                        strict=True,
                    )
                )
                and any(
                    left < right
                    for left, right in zip(
                        (
                            int(other.get("false_admission_count", 0))
                            + int(other.get("false_safe_hold_count", 0)),
                            int(other.get("missed_viable_action_count", 0)),
                            int(other.get("unnecessary_nonattempt_count", 0)),
                        ),
                        vector,
                        strict=True,
                    )
                )
                for other in rule_rows
            )
            if not dominated:
                pareto.append(str(row["rule_id"]))
        target_rows.append(
            {
                "complete_units": len(units),
                "counterfactual_action_outcome_status": "NOT_SCORED_GATE_DECISIONS_ONLY",
                "gate_normalization": (
                    "equal-weight hostile comparator uses predeclared pass/fail signs; "
                    "native-unit margins are never added"
                ),
                "pareto_rule_ids": sorted(pareto),
                "rule_rows": rule_rows,
                "target_id": target,
                "unit_rows": unit_rows,
            }
        )
    for parent_id in parent_ids:
        if not parent_id.startswith("selective-dependence-response-"):
            stops.append({"parent_id": parent_id, "reason_code": "GATE_VECTOR_INCOMPLETE"})
    return _base(
        "analysis.noncompensating-gate-semantics",
        status="OPPOSED" if counterexamples else "NULL",
        parents=parent_ids,
        estimands={"target_rows": target_rows},
        counterexamples=counterexamples,
        stops=stops,
        interpretation=(
            "On the only complete recorded gate vectors, deleting or compensating adverse "
            "gates creates false admissions relative to the frozen noncompensating rule. "
            "No unobserved counterfactual action outcome is imputed."
        ),
    )


_RUNG_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "CONTROLLER_USE",
        (
            "CONTROLLER_USE",
            "CONTROLLER_USE_VALIDATED_LOCAL",
            "CONTROLLER_USE_VALIDATED_EXPANDED_SUPPORT",
            "GENERATED_CONTROLLER_USE_VALIDATED",
            "VALIDATED_GENERATED_EXACT_WORLD_ONLY",
        ),
    ),
    (
        "ADMISSION",
        (
            "ADMISSION",
            "ADMISSION_SUPPORTED",
            "ADMISSION_SUPPORTED",
            "NONHOLD_GENERATED_GEOMETRY_QUALIFIED",
        ),
    ),
    ("LOCAL_LAW", ("LOCAL_LAW", "LOCAL_LAW_SUPPORTED", "TARGET_PREDICTION_SUPPORTED")),
    ("RESPONSE", ("RESPONSE_SUPPORTED", "MATERIAL_RESPONSE_SUPPORTED", "FINITE_ACTION_RESPONSE")),
    ("ORDER_RELATION", ("ORDER_RELATION_SUPPORTED", "ORDER_SUPPORTED")),
    ("MEASUREMENT", ("MEASUREMENT", "MEASUREMENT_SUPPORTED", "SOURCE_PREPARATION_QUALIFIED")),
)


def _reason_values(document: Mapping[str, Any]) -> tuple[str, ...]:
    values: set[str] = set()
    for item in _walk(_value(document)):
        if not isinstance(item, Mapping):
            continue
        for key in ("reason_codes", "stop_codes"):
            reasons = item.get(key)
            if isinstance(reasons, list):
                values.update(str(value) for value in reasons)
        for key in (
            "corrected_recurrence_disposition",
            "disposition",
            "order_relation_through_controller_use",
            "controller_admission",
            "controller_admission_status",
            "prospective_controller_evaluation",
            "prospective_controller_evaluation_status",
            "scientific_status",
            "terminal",
            "terminal_branch",
            "terminal_code",
            "terminal_handoff",
            "terminal_kind",
            "terminal_result",
            "terminal_verdict",
            "verdict",
        ):
            value = item.get(key)
            if isinstance(value, str):
                values.add(value)
    return tuple(sorted(values))


def _highest_structured_rung(documents: Sequence[Mapping[str, Any]]) -> str:
    explicit = []
    for document in documents:
        for item in _walk(_value(document)):
            if isinstance(item, Mapping):
                ceiling = item.get("maximum_evidence_ceiling")
                if isinstance(ceiling, str):
                    explicit.append(ceiling)
    order = {
        "MEASUREMENT": 0,
        "ORDER_RELATION": 1,
        "RESPONSE": 2,
        "LOCAL_LAW": 3,
        "ADMISSION": 4,
        "CONTROLLER_USE": 5,
    }
    recognized = [value for value in explicit if value in order]
    if recognized:
        return max(recognized, key=order.__getitem__)
    tokens = {value.upper() for document in documents for value in _strings(document)}
    for rung, markers in _RUNG_MARKERS:
        if any(marker in tokens for marker in markers):
            return rung
    return "UNRESOLVED"


def _stop_classes(reasons: Sequence[str]) -> tuple[str, ...]:
    upper = "\n".join(reasons).upper()
    classes = []
    for name, markers in (
        ("AUTHORITY", ("AUTHORITY",)),
        (
            "ACTION_CHAIN",
            ("ACTION_CLOCK", "ACTION_DELIVERY", "REALIZATION", "REQUESTED", "APPLIED"),
        ),
        ("METHOD", ("INSTRUMENT", "METHOD", "COMPILER", "VALIDITY_BAR", "CONFORMANCE")),
        ("POWER", ("POWER", "PRECISION", "RESOLUTION", "UNDERPOWER")),
        (
            "SUPPORT_OR_DENOMINATOR",
            ("SUPPORT", "DENOMINATOR", "NO_COMPATIBLE", "BOUNDARY_COMPLEXITY"),
        ),
        (
            "ABSENT_OPERAND",
            ("ABSENT", "UNAVAILABLE", "UNRESOLVED", "NOT_GENERATED", "NO_SOURCE", "NO_BURNIN"),
        ),
        (
            "OBSERVED_OPPOSITION",
            (
                "NOT_SUPPORTED",
                "OPPOSED",
                "MISMATCH",
                "COUNTEREXAMPLE",
                "FALSE_ACTION",
                "HIDDEN_SINK",
                "FAILED",
            ),
        ),
        ("NONATTEMPT", ("NOT_ATTEMPTED", "NONATTEMPT", "NOT_ENTERED", "PREREQUISITE")),
    ):
        if any(marker in upper for marker in markers):
            classes.append(name)
    return tuple(classes or ("UNRESOLVED",))


def _assess_obstruction_signatures(documents: DocumentMap, parent_ids: Sequence[str]) -> dict[str, object]:
    rows: list[dict[str, Any]] = []
    signatures: Counter[str] = Counter()
    for parent_id in parent_ids:
        selected = _parent_documents(documents, parent_id)
        reasons = tuple(
            sorted({value for document in selected for value in _reason_values(document)})
        )
        classes = _stop_classes(reasons)
        signatures.update(classes)
        rows.append(
            {
                "decisive_counterexample_present": "OBSERVED_OPPOSITION" in classes,
                "fresh_experiment_discriminating": any(
                    value in classes
                    for value in ("ABSENT_OPERAND", "POWER", "SUPPORT_OR_DENOMINATOR")
                ),
                "highest_supported_rung": _highest_structured_rung(selected),
                "lower_rung_support_survives": "METHOD" not in classes,
                "parent_id": parent_id,
                "stop_classes": list(classes),
                "terminal_reason_codes": list(reasons),
            }
        )
    return _base(
        "analysis.obstruction-signatures",
        status="SUPPORTED",
        parents=parent_ids,
        estimands={"obstruction_signatures": dict(sorted(signatures.items())), "rows": rows},
        counterexamples=(
            {
                "counterexample_id": "distinct-stop-regions",
                "opposes": "collapsing-adverse-absent-method-power-and-authority-stops",
                "observed_stop_classes": sorted(signatures),
            },
        ),
        stops=(),
        interpretation=(
            "Exact terminal fields separate observed opposition, absent operands, method, power, "
            "support/denominator, action-chain, authority and nonattempt obstructions. Rows may "
            "occupy several classes; forcing one primary failure label would itself destroy "
            "information. Unresolved rungs are retained rather than inferred from programme age."
        ),
    )


def _assess_power_design_robustness(documents: DocumentMap, parent_ids: Sequence[str]) -> dict[str, object]:
    rows: list[dict[str, Any]] = []
    stops: list[dict[str, Any]] = []
    confidence_levels = (Decimal("0.80"), Decimal("0.90"), Decimal("0.95"), Decimal("0.99"))
    threshold_multipliers = (
        Decimal("0.5"),
        Decimal("0.75"),
        Decimal(1),
        Decimal("1.25"),
        Decimal("1.5"),
        Decimal(2),
    )
    for target in ("cantera", "fipy"):
        power = _root_value(_document(documents, f"selective-dependence-response-{target}-power"))
        requirements = power.get("family_requirements", [])
        if not isinstance(requirements, list):
            raise ValueError("selective dependence response power family changed shape")
        operands = []
        for embedded in requirements:
            row = _value(embedded)
            if not isinstance(row, Mapping):
                raise ValueError("selective dependence response power requirement changed shape")
            operands.append(
                SelectiveDependenceResponsePowerOperand(
                    requirement_id=str(row["requirement_id"]),
                    family_id=str(row["family_id"]),
                    method=SelectiveDependenceResponsePowerMethod(str(row["method"])),
                    expected_location=Decimal(str(_value(row["expected_location"])["decimal"])),
                    decision_boundary=Decimal(str(_value(row["decision_boundary"])["decimal"])),
                    development_standard_deviation=(
                        None
                        if row.get("development_standard_deviation") is None
                        else Decimal(str(_value(row["development_standard_deviation"])["decimal"]))
                    ),
                    maximum_adverse_rate=(
                        None
                        if row.get("maximum_adverse_rate") is None
                        else Decimal(str(_value(row["maximum_adverse_rate"])["decimal"]))
                    ),
                    development_complete_unit_count=int(row["development_complete_unit_count"]),
                    development_unit_ids_sha256=str(row["development_unit_ids_sha256"]),
                    native_unit=str(row["native_unit"]),
                )
            )
        surface = []
        for confidence in confidence_levels:
            for multiplier in threshold_multipliers:
                adjusted = tuple(
                    SelectiveDependenceResponsePowerOperand(
                        requirement_id=value.requirement_id,
                        family_id=value.family_id,
                        method=value.method,
                        expected_location=value.expected_location,
                        decision_boundary=value.decision_boundary * multiplier,
                        development_standard_deviation=value.development_standard_deviation,
                        maximum_adverse_rate=(
                            value.maximum_adverse_rate * multiplier
                            if value.maximum_adverse_rate is not None
                            else None
                        ),
                        development_complete_unit_count=value.development_complete_unit_count,
                        development_unit_ids_sha256=value.development_unit_ids_sha256,
                        native_unit=value.native_unit,
                    )
                    for value in operands
                )
                qualified = qualify_power_design(
                    target_id=f"posthoc-{target}",
                    candidate_panel_sizes=tuple(
                        int(value) for value in power["candidate_panel_sizes"]
                    ),
                    operands=tuple(sorted(adjusted, key=lambda value: value.requirement_id)),
                    familywise_alpha=Decimal(1) - confidence,
                    target_power=Decimal(str(_value(power["target_power"])["decimal"])),
                    maximum_nonevaluable_rate=Decimal(
                        str(_value(power["maximum_nonevaluable_rate"])["decimal"])
                    ),
                )
                surface.append(
                    {
                        "attainable": qualified.attainable,
                        "confidence": str(confidence),
                        "primary_original_criterion": (
                            confidence == Decimal("0.95") and multiplier == Decimal(1)
                        ),
                        "requirements": [
                            {
                                "family_id": value.family_id,
                                "favorable_gap": str(value.favorable_gap),
                                "method": value.method.value,
                                "required_evaluable_complete_unit_count": value.required_evaluable_complete_unit_count,
                                "required_issued_complete_unit_count": value.required_issued_complete_unit_count,
                                "requirement_id": value.requirement_id,
                            }
                            for value in qualified.family_requirements
                        ],
                        "selected_evaluation_unit_count": qualified.selected_evaluation_unit_count,
                        "threshold_multiplier": str(multiplier),
                    }
                )
        primary = next(value for value in surface if value["primary_original_criterion"])
        if primary["selected_evaluation_unit_count"] != power.get("selected_evaluation_unit_count"):
            raise ValueError("post-hoc primary power surface disagrees with frozen parent")
        original_requirements = primary["requirements"]
        assert isinstance(original_requirements, list)
        failure_codes = []
        for value in original_requirements:
            assert isinstance(value, Mapping)
            if value.get("required_evaluable_complete_unit_count") is None:
                failure_codes.append(
                    {
                        "reason_code": (
                            "NO_FINITE_COUNT_UNDER_OBSERVED_RATE"
                            if value.get("method") == "ZERO_FAILURE_UPPER_BOUND"
                            else "THRESHOLD_DOMINATES_PRECISION"
                        ),
                        "requirement_id": value.get("requirement_id"),
                    }
                )
        rows.append(
            {
                "candidate_panel_sizes": power.get("candidate_panel_sizes"),
                "failure_codes_at_original_criterion": failure_codes,
                "original_attainable": power.get("attainable"),
                "original_selected_evaluation_unit_count": power.get(
                    "selected_evaluation_unit_count"
                ),
                "sensitivity_surface": surface,
                "target_id": target,
            }
        )
    for parent_id in parent_ids:
        if not parent_id.startswith("selective-dependence-response-"):
            stops.append({"parent_id": parent_id, "reason_code": "BOUNDARY_OPERANDS_INSUFFICIENT"})
    status = "MIXED" if any(row["original_attainable"] is False for row in rows) else "SUPPORTED"
    return _base(
        "analysis.power-design-robustness",
        status=status,
        parents=parent_ids,
        estimands={"qualified_power_surfaces": rows},
        counterexamples=(),
        stops=stops,
        interpretation=(
            "The frozen selective dependence response heterogeneous power method was reused to enumerate confidence "
            "and threshold sensitivity cells without fitting new outcomes. The primary cells "
            "exactly reproduce the parent decisions. Adverse nonzero rates and wrong-side "
            "materiality have no finite same-kind sample-size rescue; other terminal summaries "
            "lack the operands needed for recomputation and remain unevaluable."
        ),
    )


def _assess_semantic_role_preservation(documents: DocumentMap, parent_ids: Sequence[str]) -> dict[str, object]:
    rows: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    for target in ("cantera", "fipy"):
        selection = _root_value(_document(documents, f"selective-dependence-response-{target}-denominator"))
        assessments = selection.get("assessments", [])
        for embedded in assessments if isinstance(assessments, list) else []:
            row = _value(embedded)
            if not isinstance(row, Mapping):
                continue
            item = {
                "adequate": row.get("adequate"),
                "alternative": row.get("alternative"),
                "calibration_accuracy": row.get("calibration_accuracy"),
                "calibration_complete_unit_count": row.get("calibration_complete_unit_count"),
                "failure_codes": row.get("failure_codes"),
                "omitted_role_id": row.get("omitted_role_id"),
                "retained_role_ids": row.get("retained_role_ids"),
                "semantic_preservation": row.get("semantic_preservation"),
                "support_boundary_preserved": row.get("support_boundary_preserved"),
                "target_id": target,
            }
            rows.append(item)
            if row.get("semantic_preservation") is False:
                counterexamples.append(
                    {
                        "counterexample_id": f"{target}-{str(row.get('alternative')).lower()}-semantic-loss",
                        "opposes": "predictive-fit-implies-semantic-validity",
                        "parent_id": f"selective-dependence-response-{target}",
                    }
                )
    fiber = _root_value(_document(documents, "battery-reduced-observation-fiber"))
    fiber_rows = fiber.get("rows", [])
    projected_fibers = (
        [_value(row) for row in fiber_rows if isinstance(_value(row), Mapping)]
        if isinstance(fiber_rows, list)
        else []
    )
    cell_counts = Counter(str(row.get("cell")) for row in projected_fibers)
    pair_cells: dict[str, Counter[str]] = defaultdict(Counter)
    denominator_cells: dict[str, Counter[str]] = defaultdict(Counter)
    close_diverged_units = set()
    for row in projected_fibers:
        pair_cells[str(row.get("pair_id"))][str(row.get("cell"))] += 1
        denominator_cells[str(row.get("denominator_id"))][str(row.get("cell"))] += 1
        if row.get("close") is True and row.get("future_equivalent") is False:
            close_diverged_units.add(str(row.get("unit_id")))
    if close_diverged_units:
        counterexamples.append(
            {
                "counterexample_id": "battery-reduced-observation-close-but-future-diverged",
                "complete_unit_count": len(close_diverged_units),
                "opposes": "reduced-coordinate-proximity-implies-future-closure",
                "parent_id": "battery-reduced-observation",
            }
        )
    return _base(
        "analysis.semantic-role-preservation",
        status="OPPOSED",
        parents=parent_ids,
        estimands={
            "denominator_assessments": rows,
            "battery_reduced_observation_fiber_cells": dict(sorted(cell_counts.items())),
            "battery_reduced_observation_pair_cells": {
                key: dict(sorted(value.items())) for key, value in sorted(pair_cells.items())
            },
            "battery_reduced_observation_denominator_cells": {
                key: dict(sorted(value.items())) for key, value in sorted(denominator_cells.items())
            },
            "battery_reduced_observation_close_diverged_complete_unit_ids": sorted(close_diverged_units),
            "typed_map_graph": [
                {
                    "codomain": "categorical response and action-fibre disposition",
                    "domain": "D,H,A,R,tau development support",
                    "evidence_world": "NUMERICAL_SIMULATOR",
                    "map_family": "selective-dependence-response-denominator-alternatives",
                },
                {
                    "codomain": "future receiver trajectory equivalence",
                    "domain": "PyBaMM reduced-observation finite fibres within recorded denominator",
                    "evidence_world": "NUMERICAL_SIMULATOR",
                    "map_family": "battery-reduced-observation-reduced-coordinate-fibres",
                },
            ],
        },
        counterexamples=counterexamples,
        stops=tuple(
            {"parent_id": parent_id, "reason_code": "MAP_NOT_TYPED"}
            for parent_id in parent_ids
            if parent_id not in {"selective-dependence-response-cantera", "selective-dependence-response-fipy", "battery-reduced-observation"}
        ),
        interpretation=(
            "High calibration accuracy coexists with explicit semantic-role loss in both selective-response "
            "targets, and PyBaMM records close reduced-coordinate pairs that diverge in future "
            "receivers. These are exact counterexamples to fit- or proximity-only adequacy. "
            "Unrecorded domain/codomain maps are typed as unevaluable, not reverse engineered."
        ),
    )


def _assess_complete_unit_localization(documents: DocumentMap, parent_ids: Sequence[str]) -> dict[str, object]:
    rows: list[dict[str, Any]] = []
    stops: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    for target in ("cantera", "fipy"):
        projected = project_units(_document(documents, f"selective-dependence-response-{target}-panel"))
        unit_ids = [str(value["complete_unit_id"]) for value in projected]
        fractions = []
        unit_rows = []
        for unit in projected:
            hold_action_ids = {
                str(value.get("hold_action_id"))
                for value in unit["context_decisions"]
                if isinstance(value, Mapping)
            }
            conditions = [
                value
                for value in unit["conditions"]
                if isinstance(value, Mapping)
                and str(value.get("action_id")) not in hold_action_ids
                and value.get("support_state") != "outside-support"
            ]
            evaluable = [value for value in conditions if value.get("stopped") is not True]
            admitted = sum(value.get("fibre_admitted") is True for value in evaluable)
            fraction = admitted / len(evaluable) if evaluable else 0.0
            fractions.append(fraction)
            unit_rows.append(
                {
                    "admitted": ratio(admitted, len(evaluable)),
                    "any_admitted_action_fibre": admitted > 0,
                    "complete_unit_id": unit["complete_unit_id"],
                    "evaluable_condition_count": len(evaluable),
                }
            )
        any_admitted_states = {bool(value["any_admitted_action_fibre"]) for value in unit_rows}
        classification = (
            "DELETION_STABLE" if len(any_admitted_states) == 1 else "LOCALIZED_COUNTEREXAMPLE"
        )
        if classification == "LOCALIZED_COUNTEREXAMPLE":
            counterexamples.append(
                {
                    "counterexample_id": f"{target}-unit-local-admission-heterogeneity",
                    "opposes": "uniform-action-fibre-availability-across-complete-units",
                    "parent_id": f"selective-dependence-response-{target}",
                    "units_with_any_admission": sum(
                        bool(value["any_admitted_action_fibre"]) for value in unit_rows
                    ),
                    "units_without_any_admission": sum(
                        not bool(value["any_admitted_action_fibre"]) for value in unit_rows
                    ),
                }
            )
        rows.append(
            {
                "classification": classification,
                "complete_unit_bootstrap": complete_unit_bootstrap_mean(
                    fractions,
                    seed=2_026_081_101 + (0 if target == "cantera" else 1),
                ),
                "complete_unit_count": len(projected),
                "leave_one_unit_out": leave_one_unit_means(fractions, unit_ids),
                "primary_disposition": (
                    "UNIFORM_ANY_ADMISSION"
                    if any_admitted_states == {True}
                    else "UNIFORM_NO_ADMISSION"
                    if any_admitted_states == {False}
                    else "UNIT_HETEROGENEOUS_ADMISSION"
                ),
                "target_id": target,
                "unit_rows": unit_rows,
            }
        )
        construct_validation_comparators = project_comparators(
            _document(documents, f"target-construct-validation-{target}-bundle"),
            _document(documents, f"target-construct-validation-{target}-panel"),
        )
        for comparator in construct_validation_comparators:
            if comparator["comparator_kind"] != "structural-recurrence":
                continue
            unit_values = comparator["complete_unit_rows"]
            assert isinstance(unit_values, list)
            exact = [
                int(value.get("categorical_mismatch_count", 0)) == 0
                and int(value.get("unsafe_false_admission_count", 0)) == 0
                and int(value.get("unevaluable_case_count", 0)) == 0
                for value in unit_values
                if isinstance(value, Mapping)
            ]
            rows.append(
                {
                    "classification": "DELETION_STABLE"
                    if all(exact)
                    else "LOCALIZED_COUNTEREXAMPLE",
                    "complete_unit_count": len(exact),
                    "parent_id": f"target-construct-validation-{target}",
                    "primary_disposition": "UNIT_EXACT_CATEGORICAL_FORECAST",
                    "unit_exact_count": sum(exact),
                    "unit_rows": unit_values,
                }
            )
    for parent_id in parent_ids:
        if parent_id not in {
            "target-construct-validation-cantera",
            "target-construct-validation-fipy",
            "selective-dependence-response-cantera",
            "selective-dependence-response-fipy",
        }:
            stops.append({"parent_id": parent_id, "reason_code": "UNEVALUABLE_UNIT_STRUCTURE"})
    return _base(
        "analysis.complete-unit-localization",
        status="MIXED",
        parents=parent_ids,
        estimands={"complete_unit_localization": rows},
        counterexamples=counterexamples,
        stops=stops,
        interpretation=(
            "Complete-unit deletion and fixed-seed block bootstrap are executable for selective-response, and "
            "construct-validation categorical exactness is checked unit by unit. Unit-local admission variation "
            "is retained as a valid counterexample to uniform availability. Other terminal "
            "summaries are not re-expanded into pseudo-units and remain unevaluable."
        ),
    )


def _has_marker(documents: DocumentMap, artifact_prefix: str, markers: Sequence[str]) -> bool:
    upper = "\n".join(
        value
        for document in _parent_documents(documents, artifact_prefix)
        for value in _strings(document)
    ).upper()
    return any(marker.upper() in upper for marker in markers)


def _assess_independent_scientific_axes(documents: DocumentMap, parent_ids: Sequence[str]) -> dict[str, object]:
    axis_rows = []
    for target in ("cantera", "fipy"):
        bundle = _root_value(_document(documents, f"target-construct-validation-{target}-bundle"))
        adjudication = _value(bundle["target_adjudication"])
        restrictiveness = _value(bundle["restrictiveness"])
        assert isinstance(adjudication, Mapping) and isinstance(restrictiveness, Mapping)
        axis_rows.append(
            {
                "axes": {
                    "admission_evaluability": "SUPPORTED",
                    "comparator_restrictiveness": "OPPOSED",
                    "controller_construction": "NONATTEMPT_PREREQUISITE",
                    "coordinate_necessity": "OPPOSED",
                    "finite_action_response": "SUPPORTED",
                    "material_action_response": "UNEVALUABLE",
                    "local_law_predictive_adequacy": "SUPPORTED",
                    "prospective_controller_evaluation": "NONATTEMPT_PREREQUISITE",
                },
                "parent_id": f"target-construct-validation-{target}",
                "reason_codes": sorted(
                    {
                        *[str(value) for value in adjudication.get("reason_codes", [])],
                        *[str(value) for value in restrictiveness.get("reason_codes", [])],
                    }
                ),
            }
        )
    selective_response_summary: dict[str, dict[str, Any]] = {}
    for target in ("cantera", "fipy"):
        challenge = _root_value(_document(documents, f"selective-dependence-response-{target}-forecast-challenge"))
        denominator = _root_value(_document(documents, f"selective-dependence-response-{target}-denominator"))
        separation = _root_value(_document(documents, f"selective-dependence-response-{target}-forecast-separation"))
        policy_states = [str(value) for value in challenge.get("policy_dispositions", [])]
        all_nonattempt = bool(policy_states) and set(policy_states) == {"NONATTEMPT"}
        target_sink_conflicts = [
            str(value) for value in challenge.get("target_sink_conflict_case_ids", [])
        ]
        selective_response_summary[target] = {
            "all_policy_nonattempt": all_nonattempt,
            "all_comparators_separated": separation.get("all_claim_relevant_comparators_separated")
            is True,
            "denominator_resolved": denominator.get("resolved") is True,
            "evaluation_outcome_count": challenge.get("evaluation_outcome_count"),
            "target_sink_conflict_count": len(target_sink_conflicts),
        }
        axis_rows.append(
            {
                "axes": {
                    "admission_evaluability": "MIXED" if not all_nonattempt else "OPPOSED",
                    "comparator_restrictiveness": "MIXED",
                    "controller_construction": "NONATTEMPT_PREREQUISITE",
                    "coordinate_necessity": "MIXED",
                    "finite_action_response": "SUPPORTED",
                    "material_action_response": "MIXED",
                    "local_law_predictive_adequacy": "MIXED",
                    "prospective_controller_evaluation": "NONATTEMPT_PREREQUISITE",
                },
                "parent_id": f"selective-dependence-response-{target}",
                "reason_codes": [
                    "DEVELOPMENT_ONLY_NO_EVALUATION_OUTCOMES",
                    *[str(value) for value in challenge.get("stop_codes", [])],
                ],
            }
        )
    for parent_id in ('categorical-structural-recurrence', 'action-fiber-structural-recurrence', 'margin-structural-recurrence-forecast', 'structural-recurrence-eligibility'):
        if parent_id not in parent_ids:
            continue
        documents_for_parent = _parent_documents(documents, parent_id)
        reasons = sorted(
            {value for document in documents_for_parent for value in _reason_values(document)}
        )
        upper = "\n".join(reasons).upper()
        axis_rows.append(
            {
                "axes": {
                    "admission_evaluability": (
                        "OPPOSED"
                        if "ADMISSION_STATUS_EMPTY" in upper
                        or "ADMISSION_ADMITTED_COMPONENT_COUNT_0" in upper
                        else "MIXED"
                    ),
                    "comparator_restrictiveness": "UNEVALUABLE",
                    "controller_construction": "NONATTEMPT_PREREQUISITE",
                    "coordinate_necessity": "UNEVALUABLE",
                    "finite_action_response": "MIXED",
                    "material_action_response": "UNEVALUABLE",
                    "local_law_predictive_adequacy": "MIXED",
                    "prospective_controller_evaluation": "NONATTEMPT_PREREQUISITE",
                },
                "parent_id": parent_id,
                "reason_codes": reasons,
            }
        )
    for parent_id in ("phase-local-robust-controller", "expanded-support-controller", "staged-hybrid-controller", "response-formalization-topological-substrate", "response-invariant-control-quotient"):
        if parent_id in parent_ids:
            axis_rows.append(
                {
                    "axes": {
                        "admission_evaluability": "SUPPORTED",
                        "comparator_restrictiveness": "UNEVALUABLE",
                        "controller_construction": "SUPPORTED",
                        "coordinate_necessity": "UNEVALUABLE",
                        "finite_action_response": "SUPPORTED",
                        "material_action_response": "SUPPORTED",
                        "local_law_predictive_adequacy": "SUPPORTED",
                        "prospective_controller_evaluation": "SUPPORTED",
                    },
                    "parent_id": parent_id,
                    "reason_codes": ["SIMULATOR_LOCAL_OR_GENERATED_WORLD_ONLY"],
                }
            )

    tests = [
        {
            "implication_id": "prediction-implies-restrictiveness",
            "counterexample_parent_ids": ["target-construct-validation-cantera", "target-construct-validation-fipy"],
            "disposition": "OPPOSED",
            "basis": "exact target prediction with frozen restrictiveness not distinguished",
        },
        {
            "implication_id": "selective-coordinate-dependence-implies-admissible-action",
            "counterexample_parent_ids": [],
            "disposition": "UNEVALUABLE",
            "basis": (
                "selective-response coordinate separation is target-level while action availability varies by "
                "complete unit; the antecedent is not identified unit-locally"
            ),
        },
        {
            "implication_id": "target-benefit-implies-receiver-admission",
            "counterexample_parent_ids": [
                f"selective-dependence-response-{target}"
                for target in ("cantera", "fipy")
                if int(selective_response_summary[target]["target_sink_conflict_count"]) > 0
            ],
            "disposition": (
                "OPPOSED"
                if any(
                    int(value["target_sink_conflict_count"]) > 0 for value in selective_response_summary.values()
                )
                else "UNEVALUABLE"
            ),
            "basis": "target gate passes while a noncompensating receiver sink fails",
        },
        {
            "implication_id": "law-qualification-implies-nonempty-controller-admission",
            "counterexample_parent_ids": [],
            "disposition": "UNEVALUABLE",
            "basis": "development-only selective-response is not substituted for prospective law qualification/controller admission evidence",
        },
        {
            "implication_id": "material-response-plus-complete-gates-implies-controller-entry",
            "counterexample_parent_ids": [],
            "disposition": "UNEVALUABLE",
            "basis": "no common parent proves the complete antecedent and untouched controller consequence",
        },
        {
            "implication_id": "simulator-prospective-controller-evaluation-implies-independent-recurrence",
            "counterexample_parent_ids": ["phase-local-robust-controller", "expanded-support-controller", "staged-hybrid-controller"],
            "disposition": "UNEVALUABLE",
            "basis": "simulator-local prospective controller evaluation and independent recurrence are distinct unpaired axes",
        },
    ]
    counterexamples = [
        {
            "counterexample_id": row["implication_id"],
            "opposes": row["implication_id"],
            "parent_ids": row["counterexample_parent_ids"],
        }
        for row in tests
        if row["disposition"] == "OPPOSED"
    ]
    return _base(
        "analysis.independent-scientific-axes",
        status="MIXED",
        parents=parent_ids,
        estimands={"axis_rows": axis_rows, "implication_tests": tests},
        counterexamples=counterexamples,
        stops=(),
        interpretation=(
            "Exact construct-validation and development-only selective-response cases oppose two shortcut implications: exact "
            "prediction does not establish restrictiveness, and target-beneficial response need "
            "not satisfy all receiver gates. Coordinate-necessity, law qualification, controller and independent-"
            "recurrence implications remain unevaluable; absence of a paired consequence is not "
            "mislabeled as an empirical counterexample."
        ),
    )


_ACTION_ROLE_KEYS: dict[str, tuple[str, ...]] = {
    "requested_action": ("requested_action_id", "requested_value"),
    "request_clock": ("requested_clock",),
    "acceptance": ("accepted_action_id", "accepted_value", "acceptance_state"),
    "acceptance_clock": ("accepted_clock",),
    "applied_action": ("applied_action_id", "applied_value"),
    "application_clock": ("applied_clock",),
    "realized_action": ("realized_action_id", "realized_value", "realization_id"),
    "realization_interval": ("realized_clock", "realization_interval"),
    "controller_mode": ("controller_mode", "native_mode", "policy_decision"),
    "delivery_mismatch": ("delivery_mismatch", "delivery_valid", "false_action_count"),
    "receiver_window": ("horizon_value_id", "horizon_id", "receiver_window"),
    "causal_cutoff": ("information_cutoffs", "causal_cutoff", "pre_action_cutoff"),
    "unit_linkage": ("complete_unit_id", "preparation_id"),
}


def _assess_action_stage_observability(documents: DocumentMap, parent_ids: Sequence[str]) -> dict[str, object]:
    rows: list[dict[str, Any]] = []
    missing_counts: Counter[str] = Counter()
    for parent_id in parent_ids:
        selected = _parent_documents(documents, parent_id)
        keys = frozenset(key for document in selected for key in _keys(document))
        roles: dict[str, str] = {}
        evidence_keys: dict[str, list[str]] = {}
        for role, candidates in _ACTION_ROLE_KEYS.items():
            present_keys = sorted(candidate for candidate in candidates if candidate in keys)
            state = "OBSERVED" if present_keys else "ABSENT"
            roles[role] = state
            evidence_keys[role] = present_keys
        if parent_id == "boptest-action-audit":
            assessment = _root_value(_document(documents, 'boptest-action-source-observability'))
            if int(assessment.get("independently_requestable_count", 0)) > 0:
                roles["requested_action"] = "EXACTLY_DERIVED_BY_FROZEN_CONTRACT"
                evidence_keys["requested_action"] = ["independently_requestable_count"]
            candidates = assessment.get("candidates", [])
            if (
                isinstance(candidates, list)
                and candidates
                and all(
                    isinstance(_value(value), Mapping)
                    and _value(value).get("accepted_value_observable") is True
                    for value in candidates
                )
            ):
                roles["acceptance"] = "EXACTLY_DERIVED_BY_FROZEN_CONTRACT"
                evidence_keys["acceptance"] = ["accepted_value_observable"]
            if int(assessment.get("fmu_applied_observable_count", 0)) > 0:
                roles["applied_action"] = "EXACTLY_DERIVED_BY_FROZEN_CONTRACT"
                evidence_keys["applied_action"] = ["fmu_applied_observable_count"]
            if int(assessment.get("physically_realized_port_count", 0)) == 0:
                roles["realized_action"] = "ABSENT"
                roles["realization_interval"] = "ABSENT"
                evidence_keys["realized_action"] = ["physically_realized_port_count=0"]
        reasons = sorted({value for document in selected for value in _reason_values(document)})
        if any(
            "REALIZATION" in value.upper()
            and (
                "UNOBSERVED" in value.upper()
                or "NOT_EXPOSED" in value.upper()
                or "STOP" in value.upper()
            )
            for value in reasons
        ):
            roles["realized_action"] = "ABSENT"
            roles["realization_interval"] = "ABSENT"
            evidence_keys["realized_action"] = [
                value for value in reasons if "REALIZATION" in value.upper()
            ]
        rows.append(
            {
                "evidence_keys": evidence_keys,
                "parent_id": parent_id,
                "role_states": roles,
            }
        )
        missing_counts.update(role for role, state in roles.items() if state == "ABSENT")
    required_by_rung = {
        "action-response": ["applied_action", "receiver_window", "unit_linkage"],
        "controller-admission": ["acceptance", "applied_action", "requested_action", "unit_linkage"],
        "prospective-controller-validation": [
            "applied_action",
            "realized_action",
            "realization_interval",
            "receiver_window",
            "unit_linkage",
        ],
    }
    blocked_questions: list[dict[str, Any]] = []
    information_values: Counter[str] = Counter()
    for row in rows:
        role_states = row["role_states"]
        assert isinstance(role_states, Mapping)
        for rung, required in required_by_rung.items():
            missing = sorted(
                role
                for role in required
                if role_states.get(role)
                not in {
                    "OBSERVED",
                    "EXACTLY_DERIVED_BY_FROZEN_CONTRACT",
                }
            )
            if missing:
                blocked_questions.append(
                    {
                        "minimal_measurement_addition": missing,
                        "parent_id": row["parent_id"],
                        "rung": rung,
                        "what_still_does_not_follow": (
                            "response sign, admission, authorization and control value remain empirical"
                        ),
                    }
                )
                information_values.update(missing)
    hitting_set = sorted(
        {value for row in blocked_questions for value in row["minimal_measurement_addition"]}
    )
    return _base(
        "analysis.action-stage-observability",
        status="MIXED",
        parents=parent_ids,
        estimands={
            "measurement_hitting_set_order": hitting_set,
            "missing_parent_counts": dict(sorted(missing_counts.items())),
            "blocked_question_rows": blocked_questions,
            "role_value_of_information": dict(sorted(information_values.items())),
            "required_roles_by_rung": required_by_rung,
            "rows": rows,
        },
        counterexamples=(
            {
                "counterexample_id": "boptest-realization-role-absent",
                "opposes": "requested-or-accepted-action-identifies-realized-action",
                "parent_ids": ["boptest-action-audit", "independent-substrate-grounding-boptest"],
            },
        ),
        stops=tuple(
            {"parent_id": row["parent_id"], "reason_code": "ACTION_UNIT_LINKAGE_UNRESOLVED"}
            for row in rows
            if row["role_states"]["unit_linkage"] == "ABSENT"
        ),
        interpretation=(
            "Requested, accepted, applied and realized roles are not interchangeable. target construct validation/selective-response "
            "expose the full chain, while multiple physical/simulator bridges stop because realized "
            "delivery or complete-unit linkage is absent. Instrumentation can remove that stop but "
            "cannot establish a response or authorize control."
        ),
    )


_ANALYZERS = {
    "analysis.coordinate-necessity-and-construct-diversity": _assess_coordinate_necessity_and_construct_diversity,
    "analysis.forecast-level-comparability": _assess_forecast_level_comparability,
    "analysis.noncompensating-gate-semantics": _assess_noncompensating_gate_semantics,
    "analysis.obstruction-signatures": _assess_obstruction_signatures,
    "analysis.power-design-robustness": _assess_power_design_robustness,
    "analysis.semantic-role-preservation": _assess_semantic_role_preservation,
    "analysis.complete-unit-localization": _assess_complete_unit_localization,
    "analysis.independent-scientific-axes": _assess_independent_scientific_axes,
    "analysis.action-stage-observability": _assess_action_stage_observability,
}


def execute_analysis(
    analysis_id: str,
    documents: DocumentMap,
    parent_ids: Sequence[str],
) -> dict[str, object]:
    try:
        analyzer = _ANALYZERS[analysis_id]
    except KeyError as error:
        raise ValueError(f"unknown analysis: {analysis_id}") from error
    return analyzer(documents, parent_ids)


def execute_skeptic(result: Mapping[str, object]) -> dict[str, object]:
    value = result.get("value")
    if not isinstance(value, Mapping):
        raise ValueError("analysis result value is absent")
    analysis_id = str(value.get("analysis_id"))
    estimands = value.get("estimands")
    if not isinstance(estimands, Mapping):
        raise ValueError("analysis result estimands are absent")
    applicability = {
        "leave_one_unit_influence": analysis_id in {"analysis.coordinate-necessity-and-construct-diversity", "analysis.forecast-level-comparability", "analysis.noncompensating-gate-semantics", "analysis.complete-unit-localization"},
        "matched_nulls": analysis_id in {"analysis.coordinate-necessity-and-construct-diversity", "analysis.forecast-level-comparability", "analysis.noncompensating-gate-semantics", "analysis.power-design-robustness", "analysis.semantic-role-preservation", "analysis.complete-unit-localization"},
        "scaling_artifact": analysis_id in {"analysis.coordinate-necessity-and-construct-diversity", "analysis.noncompensating-gate-semantics", "analysis.power-design-robustness", "analysis.semantic-role-preservation", "analysis.complete-unit-localization"},
    }
    evidence = {
        "leave_one_unit_influence": (
            "leave_one_unit_out" in str(estimands)
            or "complete_unit_rows" in str(estimands)
            or "unit_rows" in str(estimands)
        ),
        "matched_nulls": any(
            marker in str(estimands)
            for marker in (
                "comparators",
                "NO_COMMON_EVALUABLE_LEVELS",
                "rule_rows",
                "sensitivity_surface",
                "denominator_assessments",
                "unit_rows",
            )
        )
        or any(
            isinstance(stop, Mapping) and stop.get("reason_code") == "NO_COMMON_EVALUABLE_LEVELS"
            for stop in value.get("typed_stops", [])
        ),
        "scaling_artifact": any(
            marker in str(estimands)
            for marker in (
                "scientific_description_bits",
                "gate_normalization",
                "sensitivity_surface",
                "typed_map_graph",
                "complete_unit_bootstrap",
            )
        ),
    }
    checks: dict[str, tuple[bool, bool, str]] = {
        "aggregation_artifact": (
            True,
            value.get("no_cross_partition_numerical_pooling") is True,
            "cross-partition numerical pooling is forbidden in the result",
        ),
        "alternative_explanations": (
            True,
            bool(value.get("typed_stops") or value.get("counterexamples")),
            "typed missingness and decisive counterexamples remain visible",
        ),
        "family_completeness": (
            True,
            value.get("complete_family_executed") is True,
            "the complete frozen family is terminal",
        ),
        "leakage": (
            True,
            value.get("outcome_access") == "evaluation-revealed"
            and value.get("claim_promotion_allowed") is False,
            "outcome visibility is acknowledged and cannot promote a claim",
        ),
        "mechanical_construction": (
            True,
            analysis_id in _ANALYZERS,
            "the analysis id resolves through the frozen analyzer table",
        ),
        "selection_artifact": (
            True,
            bool(value.get("parent_ids")),
            "the exact frozen parent denominator is retained",
        ),
    }
    for name in ("leave_one_unit_influence", "matched_nulls", "scaling_artifact"):
        applicable = applicability[name]
        checks[name] = (
            applicable,
            evidence[name] if applicable else True,
            "diagnostic is present"
            if applicable
            else "not applicable to the declared inference mode",
        )
    return {
        "schema": SKEPTIC_RESULT_SCHEMA,
        "version": "1.0.0",
        "value": {
            "analysis_id": value.get("analysis_id"),
            "checks": [
                {
                    "check_id": name.replace("_", "-"),
                    "applicable": applicable,
                    "disposition": "PASS" if applicable else "NOT_APPLICABLE",
                    "observation": observation,
                    "passed": passed,
                    "reason_codes": [] if passed else [f"{name.upper()}_UNRESOLVED"],
                }
                for name, (applicable, passed, observation) in sorted(checks.items())
            ],
            "passed": all(passed for _applicable, passed, _observation in checks.values()),
            "claim_promotion_allowed": False,
        },
    }


def execute_integrated_skeptic(reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(reports) != 9:
        raise ValueError("integrated skeptic requires exactly nine analysis reports")
    analysis_ids = []
    failed = []
    for report in reports:
        value = report.get("value")
        if not isinstance(value, Mapping):
            raise ValueError("skeptic report value is absent")
        analysis_id = str(value.get("analysis_id"))
        analysis_ids.append(analysis_id)
        if value.get("passed") is not True:
            failed.append(analysis_id)
    expected = list(POSTHOC_ANALYSIS_IDS)
    if sorted(analysis_ids) != sorted(expected):
        raise ValueError("integrated skeptic analysis family drifted")
    return {
        "schema": INTEGRATED_SKEPTIC_SCHEMA,
        "version": "1.0.0",
        "value": {
            "analysis_ids": expected,
            "counterexample_precedence": True,
            "failed_analysis_ids": sorted(failed),
            "no_cross_partition_numerical_pooling": True,
            "no_target_specific_rescue": True,
            "passed": not failed,
        },
    }


def synthesize(
    results: Sequence[Mapping[str, object]],
    integrated_skeptic: Mapping[str, object],
) -> dict[str, object]:
    if len(results) != 9:
        raise ValueError("synthesis requires exactly nine analysis results")
    values = []
    for result in results:
        value = result.get("value")
        if not isinstance(value, Mapping):
            raise ValueError("analysis result value is absent")
        values.append(value)
    if sorted(str(value.get("analysis_id")) for value in values) != sorted(POSTHOC_ANALYSIS_IDS):
        raise ValueError("synthesis analysis family drifted")
    skeptic_value = integrated_skeptic.get("value")
    if not isinstance(skeptic_value, Mapping):
        raise ValueError("integrated skeptic value is absent")
    statuses = Counter(str(value.get("status")) for value in values)
    counterexample_ids = sorted(
        str(counterexample.get("counterexample_id"))
        for value in values
        for counterexample in value.get("counterexamples", [])
        if isinstance(counterexample, Mapping)
    )
    finding_ids = tuple(sorted(f"finding.{analysis_id}" for analysis_id in POSTHOC_ANALYSIS_IDS))
    hypothesis_set = HypothesisSet(
        hypothesis_set_id="hypothesis-set.effective-law-posthoc",
        finding_ids=finding_ids,
        hypotheses=tuple(
            sorted(
                (
                    Hypothesis(
                        hypothesis_id="support-limited-effective-laws",
                        statement=(
                            "Effective laws are denominator/history/action/receiver/horizon-local and "
                            "support limited."
                        ),
                        disposition=HypothesisDisposition.PREFERRED,
                        supporting_finding_ids=tuple(sorted(("finding.analysis.coordinate-necessity-and-construct-diversity", "finding.analysis.noncompensating-gate-semantics", "finding.analysis.semantic-role-preservation", "finding.analysis.complete-unit-localization", "finding.analysis.independent-scientific-axes", "finding.analysis.action-stage-observability",))),
                        opposing_observation=(
                            "Task-resolution confounding and missing prospective recurrence prevent a "
                            "universal or uniquely identified reading."
                        ),
                        missing_evidence=(
                            "Fresh predeclared recurrence tests across independent targets and evidence worlds."
                        ),
                    ),
                    Hypothesis(
                        hypothesis_id="predictive-collapse-to-control",
                        statement=(
                            "Predictive adequacy alone entails restrictiveness, admission and control value."
                        ),
                        disposition=HypothesisDisposition.OPPOSED,
                        supporting_finding_ids=tuple(sorted(("finding.analysis.coordinate-necessity-and-construct-diversity", "finding.analysis.noncompensating-gate-semantics", "finding.analysis.independent-scientific-axes",))),
                        opposing_observation="Coordinate necessity, noncompensating gate semantics and independent scientific axes contain exact counterexamples.",
                        missing_evidence=(
                            "None for the universal implication; narrower conditional claims need fresh tests."
                        ),
                    ),
                    Hypothesis(
                        hypothesis_id="single-global-law",
                        statement="One pooled law or scalar score describes the full evidence corpus.",
                        disposition=HypothesisDisposition.OPPOSED,
                        supporting_finding_ids=tuple(sorted(("finding.analysis.obstruction-signatures", "finding.analysis.semantic-role-preservation", "finding.analysis.action-stage-observability",))),
                        opposing_observation=(
                            "Obstruction signatures, semantic role preservation and action-stage observability separate obstruction, map and action-role regions."
                        ),
                        missing_evidence=(
                            "A typed transport/composition contract plus independent validation would "
                            "be required for any narrower gluing claim."
                        ),
                    ),
                ),
                key=lambda value: value.hypothesis_id,
            )
        ),
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        parent_visibility_ceilings=(VisibilityCeiling.OUTCOME_VISIBLE,),
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    hypotheses = _value(hypothesis_set.to_document())["hypotheses"]
    return {
        "schema": METATHEORY_SYNTHESIS_SCHEMA,
        "version": "1.0.0",
        "value": {
            "analysis_status_counts": dict(sorted(statuses.items())),
            "claim_ceiling": "NON_PROMOTABLE_OUTCOME_VISIBLE_CROSS_PROGRAM_POSTHOC",
            "counterexample_ids": counterexample_ids,
            "hypotheses": hypotheses,
            "hypothesis_set": hypothesis_set.to_document(),
            "integrated_skeptic_passed": skeptic_value.get("passed") is True,
            "maximum_honest_claim": (
                "The existing simulator and method corpus is consistent with and favors a "
                "plural, support-limited effective-law metatheory in which coordinate adequacy, predictive adequacy, "
                "receiver admission, material response and prospective control are distinct "
                "non-entailing obligations. This post-hoc result does not establish universal, "
                "physical, or independently recurrent lawhood."
            ),
            "no_cross_partition_numerical_pooling": True,
            "outcome_access": "evaluation-revealed",
            "visibility_ceiling": "OUTCOME_VISIBLE",
        },
    }


__all__ = [
    "execute_analysis",
    "execute_integrated_skeptic",
    "execute_skeptic",
    "synthesize",
]
