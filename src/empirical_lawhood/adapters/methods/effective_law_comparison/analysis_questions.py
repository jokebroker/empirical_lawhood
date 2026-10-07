# SPDX-License-Identifier: MPL-2.0
"""Scientific post hoc questions retained independently of a historical issuer."""

from __future__ import annotations

from empirical_lawhood.planning.posthoc import PosthocInferenceMode

_ANALYSIS_CONTRACTS = {
    "analysis.coordinate-necessity-and-construct-diversity": (
        "Which independent substrate grounding coordinates and task constructs are necessary at each endpoint?",
        (PosthocInferenceMode.WITHIN_PARTITION_UNIT, PosthocInferenceMode.CROSS_PARTITION_LOGICAL),
        ("comparator", "outcome", "unit"),
        ("NO_COMPARABLE_TASK_ENDPOINT", "SEMANTIC_ROLE_LOSS"),
        "A simpler semantic-valid representation tying the full tuple opposes necessity.",
    ),
    "analysis.forecast-level-comparability": (
        "Do categorical forecasts transport more reliably than metric or dynamical forecasts?",
        (PosthocInferenceMode.WITHIN_PARTITION_UNIT, PosthocInferenceMode.CROSS_PARTITION_LOGICAL),
        ("forecast", "outcome", "unit"),
        ("FORECAST_CHRONOLOGY_UNPROVED", "NO_COMMON_EVALUABLE_LEVELS"),
        "An unsafe categorical miss or decisive metric/dynamical miss defeats the ordering.",
    ),
    "analysis.noncompensating-gate-semantics": (
        "Which errors are created by deleting, compensating or bypassing noncompensating gates?",
        (
            PosthocInferenceMode.WITHIN_PARTITION_UNIT,
            PosthocInferenceMode.WITHIN_PARTITION_DESCRIPTIVE,
        ),
        ("action", "gate", "unit"),
        ("COUNTERFACTUAL_OUTCOME_UNOBSERVED", "GATE_VECTOR_INCOMPLETE"),
        "Any false admission or false-safe hold is decisive for the corresponding rule.",
    ),
    "analysis.obstruction-signatures": (
        "Where does each programme stop in the measurement-to-validation chain?",
        (PosthocInferenceMode.CROSS_PARTITION_LOGICAL,),
        ("terminal",),
        ("TERMINAL_RUNG_UNRESOLVED",),
        "One exact lower-to-higher-rung failure defeats a proposed implication.",
    ),
    "analysis.power-design-robustness": (
        "Which recorded boundary and power criteria were attainable inside observed support?",
        (
            PosthocInferenceMode.WITHIN_PARTITION_UNIT,
            PosthocInferenceMode.WITHIN_PARTITION_DESCRIPTIVE,
        ),
        ("power", "unit"),
        ("BOUNDARY_OPERANDS_INSUFFICIENT", "EXTRAPOLATION_REQUIRED"),
        "A criterion unattainable throughout observed support opposes same-kind resampling rescue.",
    ),
    "analysis.semantic-role-preservation": (
        "Which typed defects appear under denominator compression and map composition?",
        (
            PosthocInferenceMode.WITHIN_PARTITION_DESCRIPTIVE,
            PosthocInferenceMode.CROSS_PARTITION_LOGICAL,
        ),
        ("denominator", "map"),
        ("MAP_NOT_TYPED",),
        "Semantic or closure failure despite good predictive fit defeats fit-only adequacy.",
    ),
    "analysis.complete-unit-localization": (
        "Which conclusions are stable to deletion of a complete independent unit?",
        (PosthocInferenceMode.WITHIN_PARTITION_UNIT,),
        ("unit",),
        ("INDEPENDENT_UNIT_UNRESOLVED", "UNEVALUABLE_UNIT_STRUCTURE"),
        "A valid one-unit or predeclared-stratum reversal defeats a uniform conclusion.",
    ),
    "analysis.independent-scientific-axes": (
        "Does prediction or coordinate adequacy entail admission, value or prospective controller evaluation?",
        (PosthocInferenceMode.CROSS_PARTITION_LOGICAL,),
        ("terminal",),
        ("CONTROL_UTILITY_AXIS_UNAVAILABLE",),
        "Any exact supported antecedent with failed or unevaluable consequent defeats implication.",
    ),
    "analysis.action-stage-observability": (
        "Which action-chain roles are observed and which missing measurements block inference?",
        (PosthocInferenceMode.CROSS_PARTITION_LOGICAL,),
        ("action",),
        ("ACTION_UNIT_LINKAGE_UNRESOLVED", "ACTION_ROLE_UNOBSERVED"),
        "A realized-action claim with an absent or ambiguous role defeats action-chain identification.",
    ),
}


ANALYSIS_CONTRACTS = _ANALYSIS_CONTRACTS

__all__ = ["ANALYSIS_CONTRACTS"]
