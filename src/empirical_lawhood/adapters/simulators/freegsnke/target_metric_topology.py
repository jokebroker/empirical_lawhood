"""Prospective topology-versus-metric analysis for independent substrate grounding FreeGSNKE.

The construction is target native: development fixes cell medians and signed
three-band response classes, then evaluation compares both descriptions on the
same issued preparation roster.  Action, receiver and horizon observations are
nested within a preparation and are conjoined before that preparation
contributes one paired binary result.  D/H references remain conditional on the
other prepared stratum.  Simultaneous uncertainty resamples complete
preparations only and never pools native values across targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from hashlib import sha256
from random import Random
from statistics import median
from typing import ClassVar

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateMetricTopologyExchange
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .metric_bootstrap_inputs import FreeGsnkeMetricBootstrapInputs
from .contracts import FreeGsnkePhase
from .target_analysis import FreeGsnkePhaseReduction, FreeGsnkePreparationReduction
from .target_design import FreeGsnkeMetricTopologyDesign, FreeGsnkeMetricTopologyExchangeDesign, FreeGsnkeMetricTopologyLevelBinding, FreeGsnkePowerFreeze


_TOPOLOGY_STATES = ("NEGATIVE", "NEUTRAL", "POSITIVE")
_BACKACTION_STATUS = "PASSIVE_NUMERICAL_OBSERVATION_NO_MEASUREMENT_BACKACTION"


def _digest_ids(values: tuple[str, ...]) -> str:
    return sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()


def _classification(value: Decimal, deadband: Decimal) -> str:
    if value > deadband:
        return "POSITIVE"
    if value < -deadband:
        return "NEGATIVE"
    return "NEUTRAL"


def _boundary_margin(value: Decimal, deadband: Decimal) -> Decimal:
    state = _classification(value, deadband)
    if state == "POSITIVE":
        return value - deadband
    if state == "NEGATIVE":
        return -deadband - value
    return min(value + deadband, deadband - value)


def _binding_applies(
    unit: FreeGsnkePreparationReduction,
    binding: FreeGsnkeMetricTopologyLevelBinding,
) -> bool:
    return binding.denominator_stratum_id in {
        None,
        unit.denominator_stratum_id,
    } and binding.history_stratum_id in {None, unit.history_stratum_id}


def _signed_value(
    *,
    unit: FreeGsnkePreparationReduction,
    binding: FreeGsnkeMetricTopologyLevelBinding,
    exchange: FreeGsnkeMetricTopologyExchangeDesign,
) -> Decimal | None:
    value = next(
        (
            contrast
            for contrast in unit.contrasts
            if contrast.branch_id == binding.action_branch_id
            and contrast.coordinate_id == binding.receiver_id
            and contrast.horizon_s == binding.horizon_s
        ),
        None,
    )
    if value is None or value.native_unit != exchange.metric_native_unit:
        return None
    return exchange.response_direction * value.delta


def _cell_key(
    *,
    exchange_id: str,
    unit: FreeGsnkePreparationReduction,
    binding: FreeGsnkeMetricTopologyLevelBinding,
) -> tuple[str, str, str, str, str, str, Decimal]:
    return (
        exchange_id,
        binding.level_id,
        unit.denominator_stratum_id,
        unit.history_stratum_id,
        binding.action_branch_id,
        binding.receiver_id,
        binding.horizon_s,
    )


@dataclass(frozen=True, slots=True)
class FreeGsnkeMetricTopologyReferenceCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-metric-topology-reference-cell'

    cell_id: str
    exchange_id: str
    level_id: str
    denominator_stratum_id: str
    history_stratum_id: str
    action_branch_id: str
    receiver_id: str
    horizon_s: Decimal
    native_unit: str
    development_unit_ids: tuple[str, ...]
    metric_median: Decimal
    topology_state_id: str

    def __post_init__(self) -> None:
        for name in (
            "cell_id",
            "exchange_id",
            "level_id",
            "denominator_stratum_id",
            "history_stratum_id",
            "action_branch_id",
            "receiver_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_decimal(self.horizon_s, field_name="horizon_s", minimum=Decimal(0))
        validate_decimal(self.metric_median, field_name="metric_median")
        require_sorted_unique_strings(
            self.development_unit_ids,
            field_name="development_unit_ids",
            allow_empty=False,
        )
        if not self.native_unit:
            raise ValueError("FreeGSNKE metric reference requires a native unit")
        if self.topology_state_id not in _TOPOLOGY_STATES:
            raise ValueError("FreeGSNKE metric reference topology state differs")

    @property
    def key(self) -> tuple[str, str, str, str, str, str, Decimal]:
        return (
            self.exchange_id,
            self.level_id,
            self.denominator_stratum_id,
            self.history_stratum_id,
            self.action_branch_id,
            self.receiver_id,
            self.horizon_s,
        )


@dataclass(frozen=True, slots=True)
class FreeGsnkeMetricTopologyReferenceFreeze(CanonicalRecord):
    """Development-only target-native metric and topology reference."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-metric-topology-reference-freeze'

    freeze_id: str
    metric_topology_design: ObjectIdentity
    development_reduction: ObjectIdentity
    development_complete_unit_ids: tuple[str, ...]
    development_complete_unit_ids_sha256: str
    cells: tuple[FreeGsnkeMetricTopologyReferenceCell, ...]
    covered_exchange_ids: tuple[str, ...]
    issue_eligible: bool
    reason_codes: tuple[str, ...]
    frozen_before_prediction_issue: bool
    evaluation_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        if (
            self.metric_topology_design.object_schema != (FreeGsnkeMetricTopologyDesign.SCHEMA)
            or self.development_reduction.object_schema != FreeGsnkePhaseReduction.SCHEMA
        ):
            raise ValueError("FreeGSNKE metric reference input schema differs")
        require_sorted_unique_strings(
            self.development_complete_unit_ids,
            field_name="development_complete_unit_ids",
        )
        validate_sha256(
            self.development_complete_unit_ids_sha256,
            field_name="development_complete_unit_ids_sha256",
        )
        if self.development_complete_unit_ids_sha256 != _digest_ids(
            self.development_complete_unit_ids
        ):
            raise ValueError("FreeGSNKE metric reference complete-unit digest differs")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if len({value.key for value in self.cells}) != len(self.cells):
            raise ValueError("FreeGSNKE metric reference repeats a native cell")
        require_sorted_unique_strings(
            self.covered_exchange_ids,
            field_name="covered_exchange_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected_eligible = bool(self.development_complete_unit_ids) and not self.reason_codes
        if self.issue_eligible != expected_eligible:
            raise ValueError("FreeGSNKE metric reference eligibility is not fact-derived")
        if not self.frozen_before_prediction_issue or self.evaluation_outcome_access_count:
            raise ValueError("FreeGSNKE metric reference crossed prediction/reveal")


def freeze_freegsnke_metric_topology_reference(
    *,
    freeze_id: str,
    design: FreeGsnkeMetricTopologyDesign,
    development_reduction: FreeGsnkePhaseReduction,
) -> FreeGsnkeMetricTopologyReferenceFreeze:
    """Fit cell medians/classes on complete development preparations only."""

    if (
        development_reduction.phase is not FreeGsnkePhase.DEVELOPMENT
        or development_reduction.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
    ):
        raise ValueError("FreeGSNKE metric reference requires development-visible evidence")
    complete_units = tuple(
        value for value in development_reduction.units if value.complete_for_inference
    )
    grouped: dict[
        tuple[str, str, str, str, str, str, Decimal],
        list[tuple[str, Decimal, str]],
    ] = {}
    reasons = set(development_reduction.reason_codes)
    covered_levels: dict[str, set[str]] = {value.exchange_id: set() for value in design.exchanges}
    for exchange in design.exchanges:
        for unit in complete_units:
            for binding in exchange.level_bindings:
                if not _binding_applies(unit, binding):
                    continue
                value = _signed_value(unit=unit, binding=binding, exchange=exchange)
                if value is None:
                    reasons.add("DEVELOPMENT_METRIC_TOPOLOGY_OBSERVATION_MISSING")
                    continue
                key = _cell_key(
                    exchange_id=exchange.exchange_id,
                    unit=unit,
                    binding=binding,
                )
                grouped.setdefault(key, []).append(
                    (unit.unit_id, value, exchange.metric_native_unit)
                )
                covered_levels[exchange.exchange_id].add(binding.level_id)
    cells = []
    for index, (key, observations) in enumerate(sorted(grouped.items()), start=1):
        exchange_id, level_id, denominator_id, history_id, branch_id, receiver_id, horizon = key
        exchange = next(value for value in design.exchanges if value.exchange_id == exchange_id)
        unit_ids = tuple(sorted(value[0] for value in observations))
        metric_median = median(value[1] for value in observations)
        cells.append(
            FreeGsnkeMetricTopologyReferenceCell(
                cell_id=f"reference-cell.{freeze_id}.{index:04d}",
                exchange_id=exchange_id,
                level_id=level_id,
                denominator_stratum_id=denominator_id,
                history_stratum_id=history_id,
                action_branch_id=branch_id,
                receiver_id=receiver_id,
                horizon_s=horizon,
                native_unit=observations[0][2],
                development_unit_ids=unit_ids,
                metric_median=metric_median,
                topology_state_id=_classification(
                    metric_median,
                    exchange.topology_deadband,
                ),
            )
        )
    covered_exchange_ids = tuple(
        sorted(
            exchange.exchange_id
            for exchange in design.exchanges
            if covered_levels[exchange.exchange_id] == set(exchange.contrast_level_ids)
        )
    )
    if set(covered_exchange_ids) != {value.exchange_id for value in design.exchanges}:
        reasons.add("DEVELOPMENT_METRIC_TOPOLOGY_LEVEL_UNCOVERED")
    if development_reduction.panel_envelope_limited:
        reasons.add("PANEL_ENVELOPE_LIMITED")
    complete_ids = tuple(value.unit_id for value in complete_units)
    reason_codes = tuple(sorted(reasons))
    return FreeGsnkeMetricTopologyReferenceFreeze(
        freeze_id=freeze_id,
        metric_topology_design=ObjectIdentity.from_record(design.design_id, design),
        development_reduction=ObjectIdentity.from_record(
            development_reduction.reduction_id,
            development_reduction,
        ),
        development_complete_unit_ids=complete_ids,
        development_complete_unit_ids_sha256=_digest_ids(complete_ids),
        cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
        covered_exchange_ids=covered_exchange_ids,
        issue_eligible=bool(complete_ids) and not reason_codes,
        reason_codes=reason_codes,
        frozen_before_prediction_issue=True,
        evaluation_outcome_access_count=0,
    )


@dataclass(frozen=True, slots=True)
class FreeGsnkeMetricTopologyUnitComparison(CanonicalRecord):
    """One preparation-level paired topology/metric result."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-metric-topology-unit-comparison'

    comparison_id: str
    exchange_id: str
    unit_id: str
    applicable_level_ids: tuple[str, ...]
    nested_observation_count: int
    topology_success: bool
    metric_success: bool
    physical_response_margin: Decimal
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("comparison_id", "exchange_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.applicable_level_ids,
            field_name="applicable_level_ids",
        )
        if self.nested_observation_count < 0:
            raise ValueError("FreeGSNKE nested observation count must be nonnegative")
        validate_decimal(
            self.physical_response_margin,
            field_name="physical_response_margin",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


def _bootstrap_intervals(
    *,
    bootstrap_inputs: FreeGsnkeMetricBootstrapInputs | None,
    reference: FreeGsnkeMetricTopologyReferenceFreeze,
    evaluation_reduction: FreeGsnkePhaseReduction,
    exchange_ids: tuple[str, ...],
    comparisons: tuple[FreeGsnkeMetricTopologyUnitComparison, ...],
    replications: int,
    familywise_alpha: Decimal,
) -> dict[str, tuple[Decimal, Decimal]]:
    unit_ids = tuple(value.unit_id for value in evaluation_reduction.units)
    by_exchange_unit = {(value.exchange_id, value.unit_id): value for value in comparisons}
    if type(bootstrap_inputs) is not FreeGsnkeMetricBootstrapInputs:
        raise ValueError("FreeGSNKE metric bootstrap requires separately verified full numerical inputs and current export custody before resampling")
    bootstrap_inputs.validate_census(
        reference=ObjectIdentity.from_record(reference.freeze_id, reference),
        evaluation=ObjectIdentity.from_record(evaluation_reduction.reduction_id, evaluation_reduction),
        unit_ids=unit_ids,
        exchange_ids=exchange_ids,
        replications=replications,
        familywise_alpha=familywise_alpha,
    )
    rng = Random(bootstrap_inputs.full_seed)
    samples: dict[str, list[Decimal]] = {value: [] for value in exchange_ids}
    unit_count = len(unit_ids)
    for _ in range(replications):
        indices = tuple(rng.randrange(unit_count) for _ in range(unit_count))
        for exchange_id in exchange_ids:
            paired = tuple(
                int(by_exchange_unit[(exchange_id, unit_ids[index])].topology_success)
                - int(by_exchange_unit[(exchange_id, unit_ids[index])].metric_success)
                for index in indices
            )
            samples[exchange_id].append(Decimal(sum(paired)) / Decimal(unit_count))
    tail = familywise_alpha / (Decimal(2) * Decimal(len(exchange_ids)))
    lower_index = int((tail * Decimal(replications - 1)).to_integral_value(rounding=ROUND_FLOOR))
    upper_index = int(
        ((Decimal(1) - tail) * Decimal(replications - 1)).to_integral_value(rounding=ROUND_CEILING)
    )
    return {
        exchange_id: (
            sorted(values)[lower_index],
            sorted(values)[upper_index],
        )
        for exchange_id, values in samples.items()
    }


@dataclass(frozen=True, slots=True)
class FreeGsnkeMetricTopologyAnalysis(CanonicalRecord):
    """Five-exchange complete-unit analysis with simultaneous uncertainty."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-metric-topology-analysis'

    analysis_id: str
    metric_topology_design: ObjectIdentity
    reference_freeze: FreeGsnkeMetricTopologyReferenceFreeze
    evaluation_reduction: FreeGsnkePhaseReduction
    power_freeze: FreeGsnkePowerFreeze
    unit_comparisons: tuple[FreeGsnkeMetricTopologyUnitComparison, ...]
    exchanges: tuple[IndependentSubstrateMetricTopologyExchange, ...]
    familywise_alpha: Decimal
    bootstrap_replications: int
    complete_unit_resampling: bool
    nested_observations_count_as_units: bool
    bootstrap_inputs: FreeGsnkeMetricBootstrapInputs | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.analysis_id, field_name="analysis_id")
        if self.metric_topology_design != self.reference_freeze.metric_topology_design:
            raise ValueError("FreeGSNKE metric analysis design/reference differs")
        if self.reference_freeze.development_reduction == ObjectIdentity.from_record(
            self.evaluation_reduction.reduction_id,
            self.evaluation_reduction,
        ):
            raise ValueError("FreeGSNKE metric analysis reused development as evaluation")
        if (
            self.evaluation_reduction.phase is not FreeGsnkePhase.EVALUATION
            or self.evaluation_reduction.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or self.power_freeze.evaluation_roster != self.evaluation_reduction.phase_roster
        ):
            raise ValueError("FreeGSNKE metric analysis evaluation roster differs")
        require_sorted_unique_ids(
            self.unit_comparisons,
            attribute="comparison_id",
            field_name="unit_comparisons",
        )
        require_sorted_unique_ids(self.exchanges, attribute="exchange_id", field_name="exchanges")
        exchange_ids = tuple(value.exchange_id for value in self.exchanges)
        unit_ids = tuple(value.unit_id for value in self.evaluation_reduction.units)
        if {(value.exchange_id, value.unit_id) for value in self.unit_comparisons} != {
            (exchange_id, unit_id) for exchange_id in exchange_ids for unit_id in unit_ids
        }:
            raise ValueError("FreeGSNKE metric analysis changed the exchange/unit product")
        by_exchange = {
            exchange_id: tuple(
                value for value in self.unit_comparisons if value.exchange_id == exchange_id
            )
            for exchange_id in exchange_ids
        }
        expected_intervals = _bootstrap_intervals(
            bootstrap_inputs=self.bootstrap_inputs,
            reference=self.reference_freeze,
            evaluation_reduction=self.evaluation_reduction,
            exchange_ids=exchange_ids,
            comparisons=self.unit_comparisons,
            replications=self.bootstrap_replications,
            familywise_alpha=self.familywise_alpha,
        )
        for exchange in self.exchanges:
            values = by_exchange[exchange.exchange_id]
            expected = (
                sum(value.topology_success for value in values),
                sum(value.metric_success for value in values),
                len(unit_ids),
                min(value.physical_response_margin for value in values),
                *expected_intervals[exchange.exchange_id],
            )
            observed = (
                exchange.topology_success_count,
                exchange.metric_success_count,
                exchange.complete_unit_count,
                exchange.physical_response_margin,
                exchange.simultaneous_lower,
                exchange.simultaneous_upper,
            )
            if observed != expected or exchange.certification_margin != (
                exchange.simultaneous_lower
            ):
                raise ValueError("FreeGSNKE metric exchange is not unit/bootstrap-derived")
            if (
                exchange.complete_unit_ids_sha256
                != (self.evaluation_reduction.issued_unit_ids_sha256)
                or exchange.measurement_backaction_status != _BACKACTION_STATUS
            ):
                raise ValueError("FreeGSNKE metric exchange denominator/backaction differs")
        validate_decimal(
            self.familywise_alpha,
            field_name="familywise_alpha",
            minimum=Decimal(0),
        )
        if (
            self.familywise_alpha != self.power_freeze.familywise_alpha
            or self.bootstrap_replications != self.power_freeze.complete_unit_bootstrap_replications
            or not self.complete_unit_resampling
            or self.nested_observations_count_as_units
        ):
            raise ValueError("FreeGSNKE metric analysis power/resampling rule differs")


def evaluate_freegsnke_metric_topology(
    *,
    analysis_id: str,
    design: FreeGsnkeMetricTopologyDesign,
    reference: FreeGsnkeMetricTopologyReferenceFreeze,
    evaluation_reduction: FreeGsnkePhaseReduction,
    power_freeze: FreeGsnkePowerFreeze,
    selected_claimed_law_operand_ids: tuple[str, ...],
    bootstrap_inputs: FreeGsnkeMetricBootstrapInputs | None = None,
) -> FreeGsnkeMetricTopologyAnalysis:
    """Evaluate paired topology/metric survival on all issued preparations."""

    if type(bootstrap_inputs) is not FreeGsnkeMetricBootstrapInputs:
        raise ValueError("FreeGSNKE metric analysis requires separately verified full numerical inputs and current export custody before work")
    if (
        reference.metric_topology_design != ObjectIdentity.from_record(design.design_id, design)
        or not reference.issue_eligible
    ):
        raise ValueError("FreeGSNKE metric/topology reference is not issue eligible")
    if (
        evaluation_reduction.phase is not FreeGsnkePhase.EVALUATION
        or evaluation_reduction.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        or power_freeze.evaluation_roster != evaluation_reduction.phase_roster
        or tuple(value.unit_id for value in evaluation_reduction.units)
        != power_freeze.evaluation_unit_ids
    ):
        raise ValueError("FreeGSNKE metric/topology evaluation roster differs")
    reference_by_key = {value.key: value for value in reference.cells}
    comparisons = []
    for exchange in design.exchanges:
        for unit in evaluation_reduction.units:
            bindings = tuple(
                value for value in exchange.level_bindings if _binding_applies(unit, value)
            )
            topology_values = []
            metric_values = []
            margins = []
            reasons = set()
            for binding in bindings:
                observed = _signed_value(unit=unit, binding=binding, exchange=exchange)
                reference_cell = reference_by_key.get(
                    _cell_key(
                        exchange_id=exchange.exchange_id,
                        unit=unit,
                        binding=binding,
                    )
                )
                if observed is None or reference_cell is None:
                    reasons.add("EVALUATION_METRIC_TOPOLOGY_OBSERVATION_OR_REFERENCE_MISSING")
                    continue
                topology_values.append(
                    _classification(observed, exchange.topology_deadband)
                    == reference_cell.topology_state_id
                )
                metric_values.append(
                    abs(observed - reference_cell.metric_median) <= exchange.metric_tolerance
                )
                margins.append(_boundary_margin(observed, exchange.topology_deadband))
            complete = unit.complete_for_inference
            if not complete:
                reasons.add("EVALUATION_PREPARATION_INCOMPLETE")
            if not bindings:
                reasons.add("EVALUATION_EXCHANGE_LEVEL_NOT_APPLICABLE")
            all_observed = len(topology_values) == len(metric_values) == len(bindings) > 0
            topology_success = complete and all_observed and all(topology_values)
            metric_success = complete and all_observed and all(metric_values)
            comparisons.append(
                FreeGsnkeMetricTopologyUnitComparison(
                    comparison_id=(
                        f"comparison.{analysis_id}.{exchange.exchange_id}.{unit.unit_id}"
                    ),
                    exchange_id=exchange.exchange_id,
                    unit_id=unit.unit_id,
                    applicable_level_ids=tuple(sorted(value.level_id for value in bindings)),
                    nested_observation_count=len(topology_values),
                    topology_success=topology_success,
                    metric_success=metric_success,
                    physical_response_margin=(
                        min(margins) if margins else -exchange.topology_deadband
                    ),
                    reason_codes=tuple(sorted(reasons)),
                )
            )
    ordered = tuple(sorted(comparisons, key=lambda value: value.comparison_id))
    exchange_ids = tuple(value.exchange_id for value in design.exchanges)
    intervals = _bootstrap_intervals(
        bootstrap_inputs=bootstrap_inputs,
        reference=reference,
        evaluation_reduction=evaluation_reduction,
        exchange_ids=exchange_ids,
        comparisons=ordered,
        replications=power_freeze.complete_unit_bootstrap_replications,
        familywise_alpha=power_freeze.familywise_alpha,
    )
    common = []
    claimed = set(selected_claimed_law_operand_ids)
    for exchange in design.exchanges:
        values = tuple(value for value in ordered if value.exchange_id == exchange.exchange_id)
        topology_count = sum(value.topology_success for value in values)
        metric_count = sum(value.metric_success for value in values)
        count = len(values)
        lower, upper = intervals[exchange.exchange_id]
        common.append(
            IndependentSubstrateMetricTopologyExchange(
                exchange_id=exchange.exchange_id,
                complete_unit_ids_sha256=evaluation_reduction.issued_unit_ids_sha256,
                topology_success_count=topology_count,
                metric_success_count=metric_count,
                complete_unit_count=count,
                delta=(Decimal(topology_count - metric_count) / Decimal(count)),
                simultaneous_lower=lower,
                simultaneous_upper=upper,
                physical_response_margin=min(value.physical_response_margin for value in values),
                certification_margin=lower,
                measurement_backaction_status=_BACKACTION_STATUS,
                claim_bearing=(exchange.claim_bearing and exchange.law_operand_id in claimed),
            )
        )
    return FreeGsnkeMetricTopologyAnalysis(
        analysis_id=analysis_id,
        metric_topology_design=ObjectIdentity.from_record(design.design_id, design),
        reference_freeze=reference,
        evaluation_reduction=evaluation_reduction,
        power_freeze=power_freeze,
        unit_comparisons=ordered,
        exchanges=tuple(sorted(common, key=lambda value: value.exchange_id)),
        familywise_alpha=power_freeze.familywise_alpha,
        bootstrap_replications=power_freeze.complete_unit_bootstrap_replications,
        complete_unit_resampling=True,
        nested_observations_count_as_units=False,
        bootstrap_inputs=bootstrap_inputs,
    )


__all__ = [
    'FreeGsnkeMetricTopologyAnalysis',
    'FreeGsnkeMetricTopologyReferenceCell',
    'FreeGsnkeMetricTopologyReferenceFreeze',
    'FreeGsnkeMetricTopologyUnitComparison',
    "evaluate_freegsnke_metric_topology",
    "freeze_freegsnke_metric_topology_reference",
]
