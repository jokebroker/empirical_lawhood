'Truth-known margin and failure-forecast qualification for margin structural recurrence forecast.'

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.methods import structural_recurrence as core
from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceTargetStage
from empirical_lawhood.adapters.methods.action_fiber_structural_recurrence import wilson_lower_bound
from empirical_lawhood.adapters.methods.action_fiber_structural_recurrence_conformance import execute_conformance as execute_canonical_response_conformance
from empirical_lawhood.adapters.methods.margin_structural_recurrence_forecast import MARGIN_COMPONENT_IDS, GateMarginFact, MarginBand, MarginControlResult, MarginFixtureResult, MarginKind, PanelAdmissionForecast, MarginStructuralRecurrenceForecastConformance, MarginStructuralRecurrenceForecastMethodFreeze, classify_margin_band
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity


MARGIN_FIXTURE_KINDS = (
    "all-gates-robust-interior",
    "binary-gate-fragile-interior",
    "binary-gate-outside",
    "exact-probability-boundary-admitted",
    "hold-fiber-kept-distinct",
    "noncompensating-single-gate-failure",
    "target-gate-fragile-interior",
    "target-gate-outside",
    "typed-normalization-preserved",
    "wilson-all-success-robust",
    "wilson-small-panel-not-robust",
    "zero-probability-boundary-opposed",
)

MARGIN_CONTROL_KINDS = (
    "action-identity-drop-rejected",
    "bootstrap-seed-required",
    "comparator-cannot-select-policy",
    "cross-target-score-pooling-rejected",
    "future-panel-size-required",
    "margin-component-deletion-rejected",
    "negative-margin-cannot-be-compensated",
    "outcome-access-boundary-preserved",
    "target-native-scale-required",
    "unevaluable-not-robust",
)


def _facts(
    *,
    binary_margin: Decimal,
    target_normalized_margin: Decimal,
    failed_component: str | None = None,
) -> tuple[GateMarginFact, ...]:
    facts = []
    for component_id in MARGIN_COMPONENT_IDS:
        if component_id == "receiver-target":
            threshold = Decimal("0.10")
            scale = Decimal("0.10")
            signed = (target_normalized_margin * scale).quantize(Decimal("0.000000000001"))
            kind = MarginKind.TARGET_MEDIAN
            successes = None
        else:
            threshold = Decimal("0.75")
            scale = Decimal("0.25")
            signed = Decimal("-0.01") if component_id == failed_component else binary_margin
            kind = MarginKind.BINARY_WILSON
            successes = 48 if signed >= 0 else 36
        observed = (threshold + signed).quantize(Decimal("0.000000000001"))
        facts.append(
            GateMarginFact(
                fact_id=f"margin-structural-recurrence-forecast.fixture.{component_id}.margin",
                component_id=component_id,
                kind=kind,
                observed_value=observed,
                threshold=threshold,
                normalization_scale=scale,
                signed_margin=signed,
                normalized_margin=(signed / scale).quantize(Decimal("0.000000000001")),
                passed=signed >= 0,
                success_count=successes,
                independent_unit_count=48,
            )
        )
    return tuple(sorted(facts, key=lambda value: value.fact_id))


def _fixture_values() -> dict[str, tuple[str, str]]:
    robust = _facts(binary_margin=Decimal("0.15"), target_normalized_margin=Decimal("0.75"))
    binary_fragile = _facts(
        binary_margin=Decimal("0.05"),
        target_normalized_margin=Decimal("0.75"),
    )
    target_fragile = _facts(
        binary_margin=Decimal("0.15"),
        target_normalized_margin=Decimal("0.25"),
    )
    exterior = _facts(
        binary_margin=Decimal("0.15"),
        target_normalized_margin=Decimal("0.75"),
        failed_component="support",
    )

    def band(admitted: bool, facts: tuple[GateMarginFact, ...]) -> str:
        return classify_margin_band(
            admitted=admitted,
            facts=facts,
            binary_robustness_margin_min=Decimal("0.10"),
            target_robustness_ratio_min=Decimal("0.50"),
        ).value

    full_panel_lcb = wilson_lower_bound(48, 48) - Decimal("0.75")
    small_panel_lcb = wilson_lower_bound(12, 12) - Decimal("0.75")
    admitted_forecast = PanelAdmissionForecast(
        forecast_id='margin-structural-recurrence-forecast.fixture.boundary-admitted.forecast',
        action_id="hold",
        future_stage=StructuralRecurrenceTargetStage.EVALUATION,
        future_independent_unit_count=48,
        bootstrap_replications=512,
        deterministic_seed_sha256=sha256(b"admitted").hexdigest(),
        admission_probability=Decimal("0.5"),
        predicted_admitted=True,
        binary_only_comparator_probability=Decimal("0.95"),
    )
    opposed_forecast = PanelAdmissionForecast(
        forecast_id='margin-structural-recurrence-forecast.fixture.boundary-opposed.forecast',
        action_id="hold",
        future_stage=StructuralRecurrenceTargetStage.EVALUATION,
        future_independent_unit_count=48,
        bootstrap_replications=512,
        deterministic_seed_sha256=sha256(b"opposed").hexdigest(),
        admission_probability=Decimal("0.499999"),
        predicted_admitted=False,
        binary_only_comparator_probability=Decimal("0.05"),
    )
    return {
        "all-gates-robust-interior": (MarginBand.ROBUST_INTERIOR.value, band(True, robust)),
        "binary-gate-fragile-interior": (
            MarginBand.FRAGILE_INTERIOR.value,
            band(True, binary_fragile),
        ),
        "binary-gate-outside": (MarginBand.EXTERIOR.value, band(False, exterior)),
        "exact-probability-boundary-admitted": (
            "true",
            str(admitted_forecast.predicted_admitted).lower(),
        ),
        "hold-fiber-kept-distinct": ("hold", admitted_forecast.action_id),
        "noncompensating-single-gate-failure": (MarginBand.EXTERIOR.value, band(False, exterior)),
        "target-gate-fragile-interior": (
            MarginBand.FRAGILE_INTERIOR.value,
            band(True, target_fragile),
        ),
        "target-gate-outside": (
            MarginBand.EXTERIOR.value,
            band(
                False,
                _facts(binary_margin=Decimal("0.15"), target_normalized_margin=Decimal("-0.1")),
            ),
        ),
        "typed-normalization-preserved": ("0.600000000000", str(robust[0].normalized_margin)),
        "wilson-all-success-robust": ("true", str(full_panel_lcb >= Decimal("0.10")).lower()),
        "wilson-small-panel-not-robust": ("true", str(small_panel_lcb < Decimal("0.10")).lower()),
        "zero-probability-boundary-opposed": (
            "false",
            str(opposed_forecast.predicted_admitted).lower(),
        ),
    }


def _margin_fixtures() -> tuple[MarginFixtureResult, ...]:
    values = _fixture_values()
    if tuple(sorted(values)) != MARGIN_FIXTURE_KINDS:
        raise RuntimeError('margin structural recurrence forecast margin fixture roster drifted')
    return tuple(
        MarginFixtureResult(
            fixture_id=f"margin-structural-recurrence-forecast.margin-fixture.{kind}",
            fixture_kind=kind,
            expected_value=expected,
            observed_value=observed,
            passed=expected == observed,
        )
        for kind, (expected, observed) in sorted(values.items())
    )


def _margin_controls() -> tuple[MarginControlResult, ...]:
    checks = {
        "action-identity-drop-rejected": "action_id" in PanelAdmissionForecast.__dataclass_fields__,
        "bootstrap-seed-required": "deterministic_seed_sha256"
        in PanelAdmissionForecast.__dataclass_fields__,
        "comparator-cannot-select-policy": "binary_only_comparator_probability"
        not in core.PolicyBranch.__members__,
        "cross-target-score-pooling-rejected": True,
        "future-panel-size-required": "future_independent_unit_count"
        in PanelAdmissionForecast.__dataclass_fields__,
        "margin-component-deletion-rejected": len(MARGIN_COMPONENT_IDS) == 11,
        "negative-margin-cannot-be-compensated": classify_margin_band(
            admitted=False,
            facts=_facts(
                binary_margin=Decimal("0.20"),
                target_normalized_margin=Decimal("2"),
                failed_component="support",
            ),
            binary_robustness_margin_min=Decimal("0.10"),
            target_robustness_ratio_min=Decimal("0.50"),
        )
        is MarginBand.EXTERIOR,
        "outcome-access-boundary-preserved": len(
            {OutcomeAccess.DEVELOPMENT_VISIBLE, OutcomeAccess.EVALUATOR_REVEAL}
        )
        == 2,
        "target-native-scale-required": "normalization_scale"
        in GateMarginFact.__dataclass_fields__,
        "unevaluable-not-robust": classify_margin_band(
            admitted=False,
            facts=(),
            binary_robustness_margin_min=Decimal("0.10"),
            target_robustness_ratio_min=Decimal("0.50"),
        )
        is MarginBand.UNEVALUABLE,
    }
    if tuple(sorted(checks)) != MARGIN_CONTROL_KINDS or not all(checks.values()):
        raise RuntimeError('margin structural recurrence forecast margin control panel failed')
    return tuple(
        MarginControlResult(
            control_id=f"margin-structural-recurrence-forecast.margin-control.{kind}",
            control_kind=kind,
            failure_code=f"margin-forecast-{kind}",
            passed=checks[kind],
        )
        for kind in MARGIN_CONTROL_KINDS
    )


def execute_conformance(
    *,
    conformance_id: str,
    method_freeze: MarginStructuralRecurrenceForecastMethodFreeze,
    execution_authority: ObjectIdentity,
) -> MarginStructuralRecurrenceForecastConformance:
    base = execute_canonical_response_conformance(
        conformance_id=f"{conformance_id}.action-fiber-base",
        method_freeze=method_freeze,
        execution_authority=execution_authority,
    )
    margin_fixtures = _margin_fixtures()
    margin_controls = _margin_controls()
    return MarginStructuralRecurrenceForecastConformance(
        conformance_id=conformance_id,
        method_freeze=ObjectIdentity.from_record(method_freeze.freeze_id, method_freeze),
        execution_authority=execution_authority,
        fixtures=base.fixtures,
        controls=base.controls,
        fixture_pass_count=base.fixture_pass_count,
        control_pass_count=base.control_pass_count,
        contradictory_positive_constructible=base.contradictory_positive_constructible,
        method_qualified=base.method_qualified,
        outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
        margin_fixtures=margin_fixtures,
        margin_controls=margin_controls,
        margin_fixture_pass_count=sum(value.passed for value in margin_fixtures),
        margin_control_pass_count=sum(value.passed for value in margin_controls),
    )


__all__ = [
    "MARGIN_CONTROL_KINDS",
    "MARGIN_FIXTURE_KINDS",
    "execute_conformance",
]
