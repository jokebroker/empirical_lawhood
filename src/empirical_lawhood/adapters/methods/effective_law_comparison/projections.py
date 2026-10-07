"""Strict, reusable projections for the heterogeneous post-hoc corpus.

The projection layer is deliberately small.  It unwraps only canonical
envelopes whose schema is already byte-qualified, validates the exact fields
used by an analysis, and retains the complete preparation as the statistical
unit.  It does not try to coerce unlike programme records into one table.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any


def unwrap(value: object) -> Any:
    """Recursively remove canonical envelopes without discarding field names."""

    if isinstance(value, Mapping) and set(value) == {"schema", "value", "version"}:
        return unwrap(value["value"])
    if isinstance(value, Mapping):
        return {str(key): unwrap(item) for key, item in value.items()}
    if isinstance(value, list):
        return [unwrap(item) for item in value]
    return value


def root_value(document: Mapping[str, object], *, schema: str | None = None) -> Mapping[str, Any]:
    if schema is not None and document.get("schema") != schema:
        raise ValueError(f"unexpected source schema: {document.get('schema')!r}")
    value = unwrap(document)
    if not isinstance(value, Mapping):
        raise ValueError("canonical source does not decode to a mapping")
    return value


def decimal_value(value: object, *, field_name: str) -> Decimal:
    unwrapped = unwrap(value)
    if isinstance(unwrapped, Mapping):
        unwrapped = unwrapped.get("decimal")
    if not isinstance(unwrapped, (str, int)):
        raise ValueError(f"{field_name} is not a canonical decimal")
    try:
        return Decimal(str(unwrapped))
    except InvalidOperation as error:
        raise ValueError(f"{field_name} is not a canonical decimal") from error


def _case_key(case: Mapping[str, Any], roles: Sequence[str]) -> tuple[str, ...]:
    fields = {
        "A": "native_action_value_id",
        "D": "denominator_value_id",
        "H": "history_value_id",
        "R": "receiver_value_id",
        "tau": "horizon_value_id",
    }
    try:
        return tuple(str(case[fields[role]]) for role in roles)
    except KeyError as error:
        raise ValueError(f"unknown or missing target construct validation dependency role: {error.args[0]}") from error


def project_comparators(
    bundle_document: Mapping[str, object],
    panel_document: Mapping[str, object],
) -> list[dict[str, Any]]:
    """Re-score every frozen construct-validation comparator and retain complete-unit summaries.

    Recomputed aggregate counts must equal the immutable parent score.  This is
    both a projection and an independent implementation check on the post-hoc
    unit decomposition.
    """

    bundle = root_value(
        bundle_document,
        schema='empirical-lawhood/methods/target-construct-validation/target-construct-validation-target-evaluation-bundle',
    )
    panel = root_value(
        panel_document,
        schema='empirical-lawhood/methods/target-construct-validation/target-construct-validation-categorical-forecast-panel',
    )
    restrictiveness = unwrap(bundle.get("restrictiveness"))
    cases = unwrap(panel.get("cases"))
    if not isinstance(restrictiveness, Mapping) or not isinstance(cases, list):
        raise ValueError("target construct validation comparator operands changed shape")
    scored = unwrap(restrictiveness.get("scored_comparators"))
    if not isinstance(scored, list):
        raise ValueError("target construct validation scored comparator family is absent")

    projected: list[dict[str, Any]] = []
    for embedded in scored:
        row = unwrap(embedded)
        if not isinstance(row, Mapping):
            raise ValueError("target construct validation scored comparator changed shape")
        encoding = unwrap(row.get("encoding"))
        score = unwrap(row.get("score"))
        if not isinstance(encoding, Mapping) or not isinstance(score, Mapping):
            raise ValueError("target construct validation comparator encoding or score is absent")
        roles = encoding.get("dependency_role_ids")
        lookup_cells = encoding.get("lookup_cells")
        if not isinstance(roles, list) or not isinstance(lookup_cells, list):
            raise ValueError("target construct validation comparator lookup surface changed shape")
        table: dict[tuple[str, ...], tuple[str, ...]] = {}
        for embedded_cell in lookup_cells:
            cell = unwrap(embedded_cell)
            if not isinstance(cell, Mapping):
                raise ValueError("target construct validation lookup cell changed shape")
            key = cell.get("key_values")
            emitted = cell.get("emitted_state_ids")
            if not isinstance(key, list) or not isinstance(emitted, list):
                raise ValueError("target construct validation lookup cell lacks key or states")
            table[tuple(str(value) for value in key)] = tuple(str(value) for value in emitted)

        by_unit: dict[str, Counter[str]] = defaultdict(Counter)
        for embedded_case in cases:
            case = unwrap(embedded_case)
            if not isinstance(case, Mapping):
                raise ValueError("target construct validation case changed shape")
            unit_id = str(case.get("complete_unit_id"))
            counts = by_unit[unit_id]
            counts["case_count"] += 1
            legal = tuple(str(value) for value in case.get("legal_state_ids", []))
            emitted = table.get(_case_key(case, [str(value) for value in roles]), legal)
            emitted = tuple(value for value in emitted if value in legal)
            counts["prediction_set_cardinality"] += len(emitted)
            if not emitted:
                counts["uncovered_case_count"] += 1
            if case.get("outcome") in {"MISSING", "STOPPED"}:
                counts["unevaluable_case_count"] += 1
                continue
            observed_states = {str(value) for value in case.get("observed_state_ids", [])}
            if not set(emitted).intersection(observed_states):
                counts["categorical_mismatch_count"] += 1
            admit_states = {str(value) for value in case.get("admit_or_act_state_ids", [])}
            if case.get("unsafe_observed") is True and set(emitted).intersection(admit_states):
                counts["unsafe_false_admission_count"] += 1

        count_fields = (
            "case_count",
            "categorical_mismatch_count",
            "uncovered_case_count",
            "unevaluable_case_count",
            "unsafe_false_admission_count",
            "prediction_set_cardinality",
        )
        aggregate = {name: sum(value[name] for value in by_unit.values()) for name in count_fields}
        for name, observed_count in aggregate.items():
            if int(score.get(name, -1)) != observed_count:
                raise ValueError(
                    f"target construct validation unit projection disagrees with frozen {name} for {encoding.get('kind')}"
                )
        projected.append(
            {
                "comparator_id": str(encoding.get("encoding_id")),
                "comparator_kind": str(encoding.get("kind")),
                "dependency_role_ids": [str(value) for value in roles],
                "coordinate_count": len(roles),
                "scientific_description_bits": int(score["scientific_description_bits"]),
                "exact": (
                    aggregate["categorical_mismatch_count"] == 0
                    and aggregate["uncovered_case_count"] == 0
                    and aggregate["unevaluable_case_count"] == 0
                    and aggregate["unsafe_false_admission_count"] == 0
                ),
                "aggregate": aggregate,
                "complete_unit_rows": [
                    {"complete_unit_id": unit_id, **dict(sorted(counts.items()))}
                    for unit_id, counts in sorted(by_unit.items())
                ],
            }
        )
    return sorted(projected, key=lambda value: str(value["comparator_kind"]))


def project_units(panel_document: Mapping[str, object]) -> list[dict[str, Any]]:
    """Project exact selective-response gate/action/context records by complete preparation."""

    panel = root_value(panel_document, schema='empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-target-panel')
    complete_units = unwrap(panel.get("complete_units"))
    if not isinstance(complete_units, list):
        raise ValueError("selective dependence response complete-unit panel is absent")
    projected: list[dict[str, Any]] = []
    for embedded_unit in complete_units:
        unit = unwrap(embedded_unit)
        if not isinstance(unit, Mapping):
            raise ValueError("selective dependence response complete unit changed shape")
        conditions = unwrap(unit.get("conditions"))
        decisions = unwrap(unit.get("context_decisions"))
        if not isinstance(conditions, list) or not isinstance(decisions, list):
            raise ValueError("selective dependence response unit lacks conditions or context decisions")
        condition_rows: list[dict[str, Any]] = []
        for embedded_condition in conditions:
            condition = unwrap(embedded_condition)
            if not isinstance(condition, Mapping):
                raise ValueError("selective dependence response condition changed shape")
            margins = unwrap(condition.get("gate_margins"))
            if not isinstance(margins, list):
                raise ValueError("selective dependence response condition gate vector is absent")
            gate_values: dict[str, str] = {}
            for embedded_margin in margins:
                margin = unwrap(embedded_margin)
                if not isinstance(margin, Mapping):
                    raise ValueError("selective dependence response gate margin changed shape")
                gate_values[str(margin.get("value_id"))] = str(
                    decimal_value(margin.get("value"), field_name="gate margin")
                )
            action = unwrap(condition.get("action"))
            if not isinstance(action, Mapping):
                raise ValueError("selective dependence response action realization is absent")
            condition_rows.append(
                {
                    "action_id": str(condition.get("action_id")),
                    "condition_id": str(condition.get("condition_id")),
                    "context_id": ".".join(
                        str(condition.get(name))
                        for name in ("denominator_id", "history_id", "horizon_id")
                    ),
                    "denominator_id": str(condition.get("denominator_id")),
                    "history_id": str(condition.get("history_id")),
                    "horizon_id": str(condition.get("horizon_id")),
                    "support_state": str(condition.get("support_state")),
                    "fibre_admitted": condition.get("fibre_admitted"),
                    "disposition": str(condition.get("disposition")),
                    "stopped": condition.get("stopped") is True,
                    "gate_values": dict(sorted(gate_values.items())),
                    "action_role_keys": sorted(str(key) for key in action),
                }
            )
        decision_rows: list[dict[str, Any]] = []
        for embedded_decision in decisions:
            decision = unwrap(embedded_decision)
            if not isinstance(decision, Mapping):
                raise ValueError("selective dependence response context decision changed shape")
            decision_rows.append(dict(decision))
        projected.append(
            {
                "complete_unit_id": str(unit.get("complete_unit_id")),
                "conditions": sorted(condition_rows, key=lambda value: str(value["condition_id"])),
                "context_decisions": sorted(
                    decision_rows,
                    key=lambda value: str(value.get("decision_id")),
                ),
            }
        )
    return sorted(projected, key=lambda value: str(value["complete_unit_id"]))


def nested_keys(value: object) -> frozenset[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        keys.update(str(key) for key in value)
        for item in value.values():
            keys.update(nested_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(nested_keys(item))
    return frozenset(keys)


__all__ = [
    "decimal_value",
    "nested_keys",
    'project_comparators',
    'project_units',
    "root_value",
    "unwrap",
]
