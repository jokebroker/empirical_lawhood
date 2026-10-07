"""Typed, outcome-blind support and opportunity instrument for quantum support conditioned identification.

Support is the noncompensating intersection of observation, context,
feature-specific opportunity, variation, and group completeness.  Numerical
nonzero is deliberately telemetry only and never an admission operand.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
import math
from typing import Mapping, Sequence, cast

import numpy as np

from .contracts import PLAN_ID, Panel, stable_json_bytes, target_weights
from .features import ALL_FEATURES, CATEGORICAL_FEATURES, COUNT_FEATURES, FEATURE_INDEX, GEOMETRY_FEATURES, PAIR_FEATURES, A_TO_B_LINKS, B_TO_A_LINKS, BOUNDARY_SITES, FeatureRow, extract_features, feature_names, fit_feature_context
from .source import EventRecord
from .receiver import site_counts


SUPPORT_SCHEMA = 'empirical-lawhood/simulators/quantum-support-conditioned-identification/coordinate-support'
SUPPORT_VERSION = "1.0.0"


class ObservationStatus(StrEnum):
    OBSERVED = "OBSERVED"
    INVALID = "INVALID"
    MISSING = "MISSING"


class CensoringStatus(StrEnum):
    NONE = "NONE"
    RIGHT_CENSORED = "RIGHT_CENSORED"


class SupportClass(StrEnum):
    COMPLETE_WINDOW_VALUE = "COMPLETE_WINDOW_VALUE"
    CENSORED_AGE = "CENSORED_AGE"
    DONOR_CONTEXT_VALUE = "DONOR_CONTEXT_VALUE"
    CENTERED_PAIR_VALUE = "CENTERED_PAIR_VALUE"
    BOUNDARY_DIRECTION_VALUE = "BOUNDARY_DIRECTION_VALUE"
    EXPLICIT_CATEGORY = "EXPLICIT_CATEGORY"


class CoordinateDisposition(StrEnum):
    RETAIN = "RETAIN"
    RETAIN_CENSORED_WITH_ADEQUATE_EXPOSURE = "RETAIN_CENSORED_WITH_ADEQUATE_EXPOSURE"
    PRUNE_ALLOWED_CONSTANT_TECHNICAL = "PRUNE_ALLOWED_CONSTANT_TECHNICAL"
    REJECT_INVALID_OBSERVATION = "REJECT_INVALID_OBSERVATION"
    REJECT_MISSING_OPERAND = "REJECT_MISSING_OPERAND"
    REJECT_CONTEXT_INADEQUATE = "REJECT_CONTEXT_INADEQUATE"
    REJECT_OPPORTUNITY_INADEQUATE = "REJECT_OPPORTUNITY_INADEQUATE"
    REJECT_VARIATION_INADEQUATE = "REJECT_VARIATION_INADEQUATE"
    REJECT_SEMANTIC_MISMATCH = "REJECT_SEMANTIC_MISMATCH"


@dataclass(frozen=True, slots=True)
class SupportFloors:
    observed_fraction: float = 0.99
    target_weight_coverage: float = 0.98
    observed_per_sector: int = 32
    opportunity_positive_per_sector: int = 16
    opportunity_neff_per_sector: float = 16.0
    opportunity_mass_per_sector: float = 32.0
    clock_slots_per_sector: int = 32
    variation_multiplier: float = 1.0e-6


@dataclass(frozen=True, slots=True)
class CoordinateSupportSpec:
    feature_id: str
    panel_group: str
    support_class: SupportClass
    native_unit: str
    native_scale_floor: float
    history_window: str
    receiver_frame: str
    zero_semantics: str
    opportunity_formula: str
    censoring_mode: str
    censoring_sentinel: float | None
    context_dependencies: tuple[str, ...]
    categorical_vocabulary: tuple[str, ...]
    sparse_exposure_gate: bool
    discrete_variation: bool
    allowed_constant_technical: bool = False
    schema: str = SUPPORT_SCHEMA
    version: str = SUPPORT_VERSION

    def document(self) -> dict[str, object]:
        document = asdict(self)
        document["support_class"] = self.support_class.value
        return document

    @property
    def identity(self) -> str:
        return sha256(stable_json_bytes(self.document())).hexdigest()


@dataclass(frozen=True, slots=True)
class CoordinateSupportObservation:
    parent_id: str
    role_id: str
    block_id: str
    sector_id: int
    fold_id: int
    feature_id: str
    spec_sha256: str
    observation_status: ObservationStatus
    physical_value: float | None
    physical_zero: bool | None
    opportunity_mass: float | None
    opportunity_positive: bool | None
    clock_slot_count: int | None
    censoring_status: CensoringStatus
    context_sha256: str | None
    donor_set_sha256: str | None
    validity_reason: str
    source_prefix_sha256: str
    plan_id: str = PLAN_ID

    def document(self) -> dict[str, object]:
        document = asdict(self)
        document["observation_status"] = self.observation_status.value
        document["censoring_status"] = self.censoring_status.value
        return document


@dataclass(frozen=True, slots=True)
class CoordinateSupportLedgerRow:
    feature_id: str
    spec_sha256: str
    disposition: CoordinateDisposition
    reason_codes: tuple[str, ...]
    telemetry: Mapping[str, object]

    def document(self) -> dict[str, object]:
        return {
            "feature_id": self.feature_id,
            "spec_sha256": self.spec_sha256,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            **dict(self.telemetry),
        }


@dataclass(frozen=True, slots=True)
class FeatureContextSupport:
    context_sha256: str
    donor_set_sha256: str
    donor_count_by_sector: Mapping[str, int]
    donor_target_weight_coverage: float
    raw_site_center: tuple[float, ...]
    reflected_site_center: tuple[float, ...]
    projection_residual: float
    target_rows_excluded: bool
    passed: bool
    reason_codes: tuple[str, ...]

    def document(self) -> dict[str, object]:
        return {
            **asdict(self),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class PanelAdmissionManifest:
    retained_panels: tuple[Panel, ...]
    retained_features: tuple[str, ...]
    rejected_features: Mapping[str, str]
    feature_order_by_panel: Mapping[str, tuple[str, ...]]
    categorical_gate_passed: bool
    group_complete: Mapping[str, bool]

    def document(self) -> dict[str, object]:
        return {
            "retained_panels": [panel.value for panel in self.retained_panels],
            "retained_features": list(self.retained_features),
            "rejected_features": dict(self.rejected_features),
            "feature_order_by_panel": {
                key: list(value) for key, value in self.feature_order_by_panel.items()
            },
            "categorical_gate_passed": self.categorical_gate_passed,
            "group_complete": dict(self.group_complete),
        }


@dataclass(frozen=True, slots=True)
class PrefixSupportCommitment:
    role_id: str
    parent_count: int
    support_spec_manifest_sha256: str
    observation_table_sha256: str
    ledger_sha256: str
    panel_admission_sha256: str
    future_fields_present: bool = False


@dataclass(frozen=True, slots=True)
class SupportAnalysis:
    specifications: tuple[CoordinateSupportSpec, ...]
    observations: tuple[CoordinateSupportObservation, ...]
    ledger_rows: tuple[CoordinateSupportLedgerRow, ...]
    admission: PanelAdmissionManifest
    support_spec_manifest_sha256: str

    @property
    def retained_panels(self) -> tuple[Panel, ...]:
        return self.admission.retained_panels

    @property
    def retained_features(self) -> tuple[str, ...]:
        return self.admission.retained_features

    @property
    def rejected_features(self) -> Mapping[str, str]:
        return self.admission.rejected_features

    @property
    def feature_order_by_panel(self) -> Mapping[str, tuple[str, ...]]:
        return self.admission.feature_order_by_panel

    @property
    def support_rows(self) -> tuple[Mapping[str, object], ...]:
        return tuple(row.document() for row in self.ledger_rows)

    @property
    def label(self) -> str:
        return (
            "NON_PROMOTABLE_OUTCOME_VISIBLE_NOMINATION"
            if self.retained_panels
            else "NON_PROMOTABLE_TYPED_GRAMMAR_NOT_ADMITTED"
        )

    @property
    def lag_coarsening(self) -> tuple[tuple[float, float], ...]:
        return ((0.0, 2.0 / 3.0), (2.0 / 3.0, 4.0))


def _spec(
    feature_id: str,
    panel_group: str,
    support_class: SupportClass,
    unit: str,
    floor: float,
    window: str,
    opportunity: str,
    *,
    sparse: bool = False,
    discrete: bool = False,
    sentinel: float | None = None,
    context: tuple[str, ...] = (),
    vocabulary: tuple[str, ...] = (),
    allowed_constant: bool = False,
) -> CoordinateSupportSpec:
    return CoordinateSupportSpec(
        feature_id=feature_id,
        panel_group=panel_group,
        support_class=support_class,
        native_unit=unit,
        native_scale_floor=floor,
        history_window=window,
        receiver_frame="A={0..5};B={6..11};A<-B",
        zero_semantics="PHYSICAL_VALUE_DIAGNOSTIC_ONLY",
        opportunity_formula=opportunity,
        censoring_mode="RIGHT_CENSORED" if sentinel is not None else "NONE",
        censoring_sentinel=sentinel,
        context_dependencies=context,
        categorical_vocabulary=vocabulary,
        sparse_exposure_gate=sparse,
        discrete_variation=discrete,
        allowed_constant_technical=allowed_constant,
    )


def coordinate_support_specs() -> tuple[CoordinateSupportSpec, ...]:
    specs: list[CoordinateSupportSpec] = []
    for name in COUNT_FEATURES:
        if name.startswith("age_last_"):
            specs.append(
                _spec(
                    name,
                    "COUNT",
                    SupportClass.CENSORED_AGE,
                    "J_xy^-1",
                    1.0 / 6.0,
                    "[176,200)",
                    f"QUALIFYING_EVENT_COUNT:{name}",
                    sentinel=24.0,
                )
            )
        else:
            specs.append(
                _spec(
                    name,
                    "COUNT",
                    SupportClass.COMPLETE_WINDOW_VALUE,
                    "event",
                    1.0,
                    (
                        "[192,200)"
                        if "delta" in name
                        else "[192,196)"
                        if name.startswith("count_previous")
                        else "[196,200)"
                    ),
                    "COMPLETE_WINDOW_UNIT_EXPOSURE",
                    discrete=True,
                    allowed_constant=name == "empty_recent",
                )
            )
    for name in GEOMETRY_FEATURES:
        specs.append(
            _spec(
                name,
                "SPATIAL_RECENCY",
                SupportClass.DONOR_CONTEXT_VALUE,
                "event^2",
                1.0,
                "[192,200)" if name.endswith("previous") else "[196,200)",
                "VALID_DONOR_CONTEXT_UNIT_EXPOSURE",
                context=("reflection_projected_site_center",),
            )
        )
    for name in PAIR_FEATURES:
        if name == "age_last_boundary_event":
            specs.append(
                _spec(
                    name,
                    "EVENT_PAIR",
                    SupportClass.CENSORED_AGE,
                    "J_xy^-1",
                    1.0 / 6.0,
                    "[180,200)",
                    "BOUNDARY_EVENT_COUNT",
                    sparse=True,
                    sentinel=20.0,
                )
            )
        elif name == "delta_boundary_direction":
            specs.append(
                _spec(
                    name,
                    "EVENT_PAIR",
                    SupportClass.BOUNDARY_DIRECTION_VALUE,
                    "pair",
                    1.0,
                    "[192,200)",
                    "EXPECTED_BIDIRECTIONAL_PHYSICAL_LINK_EXPOSURE",
                    sparse=True,
                )
            )
        else:
            lag = "SHORT" if name.endswith("_short") else "LONG"
            predicate = name.removesuffix("_short").removesuffix("_long")
            specs.append(
                _spec(
                    name,
                    "EVENT_PAIR",
                    SupportClass.CENTERED_PAIR_VALUE,
                    "pair",
                    1.0,
                    "[196,200)",
                    f"CLOCK_SLOTS_X_CONDITIONAL_MARK_PROBABILITY:{predicate}:{lag}",
                    sparse=True,
                )
            )
    for name in CATEGORICAL_FEATURES:
        specs.append(
            _spec(
                name,
                "CATEGORICAL",
                SupportClass.EXPLICIT_CATEGORY,
                "indicator",
                1.0,
                "[196,200)",
                "COMPLETE_WINDOW_UNIT_EXPOSURE",
                discrete=True,
                vocabulary=("A", "B", "none"),
            )
        )
    if len(specs) != 38 or len({spec.feature_id for spec in specs}) != 38:
        raise AssertionError("ordered quantum support conditioned identification support dictionary differs")
    return tuple(specs)


def support_spec_manifest() -> dict[str, object]:
    specs = coordinate_support_specs()
    return {
        "schema": 'empirical-lawhood/simulators/quantum-support-conditioned-identification/frozen-support-specification-manifest',
        "version": SUPPORT_VERSION,
        "value": {
            "coordinates": [spec.document() for spec in specs],
            "coordinate_sha256": [spec.identity for spec in specs],
            "coordinate_count": len(specs),
            "physical_zero_is_gate": False,
        },
    }


def evaluate_feature_context_support(
    records: Sequence[Sequence[EventRecord]],
    k_values: Sequence[int],
    unit_ids: Sequence[str],
    *,
    minimum_per_sector: int,
    target_rows_excluded: bool,
    require_target_exclusion: bool = True,
) -> FeatureContextSupport:
    """Fit and adjudicate one donor-only reflection-projected site context."""

    if not records or not (len(records) == len(k_values) == len(unit_ids)):
        raise ValueError("feature-context donor rows differ")
    weights = np.asarray(target_weights(list(k_values)), dtype=np.float64)
    recent_counts = np.asarray(
        [site_counts(record, start_time=196.0, end_time=200.0) for record in records],
        dtype=np.float64,
    )
    raw = weights @ recent_counts
    reflection = np.asarray([(5 - site) % 12 for site in range(12)], dtype=np.int64)
    projected = 0.5 * (raw + raw[reflection])
    residual = float(np.max(np.abs(projected - projected[reflection])))
    donor_counts = {str(sector): sum(value == sector for value in k_values) for sector in range(7)}
    coverage = float(weights[np.all(np.isfinite(recent_counts), axis=1)].sum())
    reasons: list[str] = []
    if min(donor_counts.values()) < minimum_per_sector:
        reasons.append("donor-sector-count-below-floor")
    if coverage < 0.98:
        reasons.append("donor-target-weight-coverage-below-floor")
    if not np.all(np.isfinite(projected)):
        reasons.append("nonfinite-reflection-projected-center")
    if residual > 1.0e-12:
        reasons.append("reflection-projection-residual-above-floor")
    if require_target_exclusion and not target_rows_excluded:
        reasons.append("target-row-not-excluded-from-donor-context")
    donor_hash = sha256(stable_json_bytes(list(unit_ids))).hexdigest()
    context_hash = sha256(
        stable_json_bytes(
            {
                "donor_set_sha256": donor_hash,
                "raw_site_center": raw.tolist(),
                "reflected_site_center": projected.tolist(),
            }
        )
    ).hexdigest()
    return FeatureContextSupport(
        context_sha256=context_hash,
        donor_set_sha256=donor_hash,
        donor_count_by_sector=donor_counts,
        donor_target_weight_coverage=coverage,
        raw_site_center=tuple(float(value) for value in raw),
        reflected_site_center=tuple(float(value) for value in projected),
        projection_residual=residual,
        target_rows_excluded=target_rows_excluded,
        passed=not reasons,
        reason_codes=tuple(reasons),
    )


def _events(record: Sequence[EventRecord], start: float, end: float) -> tuple[EventRecord, ...]:
    return tuple(event for event in record if start <= event.event_time < end)


def _predicate(feature_id: str, left: int, right: int) -> bool:
    left_a, right_a = left < 6, right < 6
    if feature_id.startswith("pair_aa_"):
        return left_a and right_a
    if feature_id.startswith("pair_bb_"):
        return not left_a and not right_a
    if feature_id.startswith("pair_ab_"):
        return left_a and not right_a
    if feature_id.startswith("pair_ba_"):
        return not left_a and right_a
    if feature_id.startswith("boundary_a_to_b_"):
        return (left, right) in A_TO_B_LINKS
    if feature_id.startswith("boundary_b_to_a_"):
        return (left, right) in B_TO_A_LINKS
    raise ValueError(f"unknown centered-pair support coordinate: {feature_id}")


def _conditional_mark_probability(
    record: Sequence[EventRecord],
    feature_id: str,
) -> float:
    counts = np.bincount(np.asarray([event.site for event in record], dtype=np.int64), minlength=12)
    total = int(counts.sum())
    if total < 2:
        return 0.0
    numerator = 0
    for left in range(12):
        for right in range(12):
            if _predicate(feature_id, left, right):
                numerator += int(counts[left]) * (int(counts[right]) - int(left == right))
    return float(numerator) / float(total * (total - 1))


def _clock_slots(record: Sequence[EventRecord], lag_start: float, lag_end: float) -> int:
    return sum(
        lag_start <= right.event_time - left.event_time < lag_end
        for index, left in enumerate(record[:-1])
        for right in record[index + 1 :]
    )


def _pair_opportunity(record: Sequence[EventRecord], feature_id: str) -> tuple[float, int]:
    recent = _events(record, 196.0, 200.0)
    lag_start, lag_end = (0.0, 2.0 / 3.0) if feature_id.endswith("_short") else (2.0 / 3.0, 4.0)
    slots = _clock_slots(recent, lag_start, lag_end)
    return slots * _conditional_mark_probability(recent, feature_id), slots


def _boundary_direction_opportunity(
    record: Sequence[EventRecord],
) -> tuple[float, int]:
    exposure = 0.0
    slots = 0
    for start, end in ((196.0, 200.0), (192.0, 196.0)):
        window = _events(record, start, end)
        local_slots = len(window) * (len(window) - 1) // 2
        probability = _conditional_mark_probability(
            window, "boundary_a_to_b_long"
        ) + _conditional_mark_probability(window, "boundary_b_to_a_long")
        exposure += local_slots * probability
        slots += local_slots
    return exposure, slots


def _source_hash(record: Sequence[EventRecord]) -> str:
    return sha256(
        stable_json_bytes([[event.event_index, event.event_time, event.site] for event in record])
    ).hexdigest()


def _opportunity(
    spec: CoordinateSupportSpec,
    record: Sequence[EventRecord],
) -> tuple[float, int | None, CensoringStatus]:
    name = spec.feature_id
    if spec.support_class in {
        SupportClass.COMPLETE_WINDOW_VALUE,
        SupportClass.DONOR_CONTEXT_VALUE,
        SupportClass.EXPLICIT_CATEGORY,
    }:
        return 1.0, None, CensoringStatus.NONE
    if spec.support_class is SupportClass.CENTERED_PAIR_VALUE:
        exposure, slots = _pair_opportunity(record, name)
        return exposure, slots, CensoringStatus.NONE
    if spec.support_class is SupportClass.BOUNDARY_DIRECTION_VALUE:
        exposure, slots = _boundary_direction_opportunity(record)
        return exposure, slots, CensoringStatus.NONE
    if name == "age_last_boundary_event":
        count = sum(event.site in BOUNDARY_SITES for event in _events(record, 180.0, 200.0))
    elif name == "age_last_any":
        count = len(_events(record, 176.0, 200.0))
    elif name == "age_last_a":
        count = sum(event.site < 6 for event in _events(record, 176.0, 200.0))
    elif name == "age_last_b":
        count = sum(event.site >= 6 for event in _events(record, 176.0, 200.0))
    else:
        raise ValueError(f"unknown censored-age support coordinate: {name}")
    return (
        float(count),
        None,
        CensoringStatus.RIGHT_CENSORED if count == 0 else CensoringStatus.NONE,
    )


def _physical_zero(value: float, spec: CoordinateSupportSpec) -> bool:
    if spec.discrete_variation:
        return value == 0.0
    tolerance = 64.0 * np.finfo(np.float64).eps * max(spec.native_scale_floor, 1.0)
    return bool(abs(value) <= tolerance)


def _required_start(spec: CoordinateSupportSpec) -> float:
    return float(spec.history_window.removeprefix("[").split(",", 1)[0])


def _feature_value(spec: CoordinateSupportSpec, row: FeatureRow) -> float:
    if spec.feature_id in FEATURE_INDEX:
        return float(row.values[FEATURE_INDEX[spec.feature_id]])
    if spec.feature_id == "last_region_B":
        return float(row.last_region == "B")
    if spec.feature_id == "last_region_none":
        return float(row.last_region == "none")
    raise ValueError(f"unknown feature in support dictionary: {spec.feature_id}")


def _effective_size(values: Sequence[float]) -> float:
    total = math.fsum(values)
    squares = math.fsum(value * value for value in values)
    return 0.0 if squares == 0.0 else total * total / squares


def _weighted_summary(
    observations: Sequence[CoordinateSupportObservation],
    weights: np.ndarray,
) -> tuple[float, float, float, float, int]:
    indices = [
        index
        for index, observation in enumerate(observations)
        if observation.observation_status is ObservationStatus.OBSERVED
        and observation.physical_value is not None
    ]
    if not indices:
        return math.nan, math.nan, math.nan, math.nan, 0
    local_weights = weights[indices]
    local_weights = local_weights / local_weights.sum()
    values = np.asarray(
        [float(cast(float, observations[index].physical_value)) for index in indices],
        dtype=np.float64,
    )
    mean = float(local_weights @ values)
    variance = float(local_weights @ ((values - mean) ** 2))
    return mean, variance, float(values.min()), float(values.max()), len(set(values.tolist()))


def _aggregate(
    spec: CoordinateSupportSpec,
    observations: Sequence[CoordinateSupportObservation],
    k_values: Sequence[int],
    weights: np.ndarray,
    floors: SupportFloors,
    *,
    context_passed: bool,
) -> CoordinateSupportLedgerRow:
    observed = np.asarray(
        [row.observation_status is ObservationStatus.OBSERVED for row in observations],
        dtype=np.bool_,
    )
    invalid = np.asarray(
        [row.observation_status is ObservationStatus.INVALID for row in observations],
        dtype=np.bool_,
    )
    missing = np.asarray(
        [row.observation_status is ObservationStatus.MISSING for row in observations],
        dtype=np.bool_,
    )
    sectors = np.asarray(k_values, dtype=np.int64)
    observed_by_sector: dict[str, int] = {}
    opportunity_positive_by_sector: dict[str, int] = {}
    opportunity_mass_by_sector: dict[str, float] = {}
    opportunity_neff_by_sector: dict[str, float] = {}
    clock_slots_by_sector: dict[str, int] = {}
    for sector in range(7):
        selected = np.flatnonzero(sectors == sector).tolist()
        observed_by_sector[str(sector)] = sum(bool(observed[index]) for index in selected)
        opportunities = [
            float(observations[index].opportunity_mass or 0.0)
            for index in selected
            if observed[index]
        ]
        opportunity_positive_by_sector[str(sector)] = sum(value > 0.0 for value in opportunities)
        opportunity_mass_by_sector[str(sector)] = math.fsum(opportunities)
        opportunity_neff_by_sector[str(sector)] = _effective_size(opportunities)
        clock_slots_by_sector[str(sector)] = sum(
            int(observations[index].clock_slot_count or 0) for index in selected if observed[index]
        )
    mean, variance, minimum, maximum, distinct = _weighted_summary(observations, weights)
    observed_fraction = float(np.mean(observed))
    observed_weight = float(weights[observed].sum())
    reasons: list[str] = []
    if missing.any():
        disposition = CoordinateDisposition.REJECT_MISSING_OPERAND
        reasons.append("required-support-operand-missing")
    elif (
        invalid.any()
        or observed_fraction < floors.observed_fraction
        or observed_weight < floors.target_weight_coverage
        or min(observed_by_sector.values()) < floors.observed_per_sector
    ):
        disposition = CoordinateDisposition.REJECT_INVALID_OBSERVATION
        reasons.append("observation-coverage-below-frozen-floor")
    elif spec.context_dependencies and not context_passed:
        disposition = CoordinateDisposition.REJECT_CONTEXT_INADEQUATE
        reasons.append("donor-context-contract-failed")
    elif spec.sparse_exposure_gate and (
        min(opportunity_positive_by_sector.values()) < floors.opportunity_positive_per_sector
        or min(opportunity_neff_by_sector.values()) < floors.opportunity_neff_per_sector
        or min(opportunity_mass_by_sector.values()) < floors.opportunity_mass_per_sector
        or (
            spec.support_class is SupportClass.CENTERED_PAIR_VALUE
            and min(clock_slots_by_sector.values()) < floors.clock_slots_per_sector
        )
    ):
        disposition = CoordinateDisposition.REJECT_OPPORTUNITY_INADEQUATE
        reasons.append("feature-specific-opportunity-below-frozen-floor")
    else:
        variation_passed = (
            True
            if spec.support_class is SupportClass.EXPLICIT_CATEGORY
            else distinct >= 2
            if spec.discrete_variation
            else math.isfinite(variance)
            and math.sqrt(max(variance, 0.0))
            >= floors.variation_multiplier * spec.native_scale_floor
            and maximum - minimum >= floors.variation_multiplier * spec.native_scale_floor
        )
        if not variation_passed and spec.allowed_constant_technical:
            disposition = CoordinateDisposition.PRUNE_ALLOWED_CONSTANT_TECHNICAL
            reasons.append("declared-empty-recent-constant-exception")
        elif not variation_passed:
            disposition = CoordinateDisposition.REJECT_VARIATION_INADEQUATE
            reasons.append("variation-below-frozen-native-scale-floor")
        elif spec.support_class is SupportClass.CENSORED_AGE and any(
            row.censoring_status is CensoringStatus.RIGHT_CENSORED for row in observations
        ):
            disposition = CoordinateDisposition.RETAIN_CENSORED_WITH_ADEQUATE_EXPOSURE
        else:
            disposition = CoordinateDisposition.RETAIN
    telemetry: dict[str, object] = {
        "support_class": spec.support_class.value,
        "panel_group": spec.panel_group,
        "n_allocated": len(observations),
        "n_observed": int(observed.sum()),
        "n_invalid": int(invalid.sum()),
        "n_missing": int(missing.sum()),
        "n_physical_zero": sum(row.physical_zero is True for row in observations),
        "n_censored": sum(
            row.censoring_status is CensoringStatus.RIGHT_CENSORED for row in observations
        ),
        "observed_fraction": observed_fraction,
        "target_weight_observation_coverage": observed_weight,
        "observed_by_sector": observed_by_sector,
        "opportunity_positive_by_sector": opportunity_positive_by_sector,
        "opportunity_mass_by_sector": opportunity_mass_by_sector,
        "opportunity_neff_by_sector": opportunity_neff_by_sector,
        "clock_slot_total_by_sector": clock_slots_by_sector,
        "weighted_mean": mean,
        "weighted_variance": variance,
        "weighted_standard_deviation": math.sqrt(max(variance, 0.0))
        if math.isfinite(variance)
        else math.nan,
        "minimum": minimum,
        "maximum": maximum,
        "exact_distinct_value_count": distinct,
        "context_contract_passed": context_passed,
        "physical_zero_used_as_gate": False,
    }
    return CoordinateSupportLedgerRow(
        feature_id=spec.feature_id,
        spec_sha256=spec.identity,
        disposition=disposition,
        reason_codes=tuple(reasons),
        telemetry=telemetry,
    )


def analyze_support(
    records: Sequence[Sequence[EventRecord]],
    k_values: Sequence[int],
    *,
    unit_ids: Sequence[str] | None = None,
    role_id: str,
    record_start: float,
    feature_rows: Sequence[FeatureRow] | None = None,
    folds: Sequence[int] | None = None,
    context_sha256: Sequence[str | None] | None = None,
    donor_set_sha256: Sequence[str | None] | None = None,
    context_passed: bool = True,
    floors: SupportFloors = SupportFloors(),
) -> SupportAnalysis:
    """Evaluate prefix-only typed support; no target/future argument exists."""

    if not records or len(records) != len(k_values):
        raise ValueError("typed support parent rows differ")
    if unit_ids is None:
        unit_ids = tuple(f"{role_id}.{index:05d}" for index in range(len(records)))
    if len(unit_ids) != len(records) or len(set(unit_ids)) != len(unit_ids):
        raise ValueError("typed support parent identities differ")
    if any(not 0 <= value <= 6 for value in k_values):
        raise ValueError("typed support preparation sector differs")
    if feature_rows is None:
        context = fit_feature_context(records, k_values)
        feature_rows = tuple(extract_features(record, context) for record in records)
        common_context_hash = sha256(
            stable_json_bytes({"site_center": context.site_center})
        ).hexdigest()
        context_sha256 = tuple(common_context_hash for _ in records)
        donor_hash = sha256(stable_json_bytes(list(unit_ids))).hexdigest()
        donor_set_sha256 = tuple(donor_hash for _ in records)
    if len(feature_rows) != len(records):
        raise ValueError("typed support feature rows differ")
    folds = tuple(0 for _ in records) if folds is None else tuple(folds)
    context_sha256 = (
        tuple(None for _ in records) if context_sha256 is None else tuple(context_sha256)
    )
    donor_set_sha256 = (
        tuple(None for _ in records) if donor_set_sha256 is None else tuple(donor_set_sha256)
    )
    if not (len(folds) == len(context_sha256) == len(donor_set_sha256) == len(records)):
        raise ValueError("typed support context identities differ")
    specs = coordinate_support_specs()
    manifest_sha = sha256(stable_json_bytes(support_spec_manifest())).hexdigest()
    observations: list[CoordinateSupportObservation] = []
    by_feature: dict[str, list[CoordinateSupportObservation]] = {
        spec.feature_id: [] for spec in specs
    }
    for parent_index, (record, row) in enumerate(zip(records, feature_rows, strict=True)):
        prefix_sha = _source_hash(record)
        for spec in specs:
            complete = record_start <= _required_start(spec)
            needs_context = bool(spec.context_dependencies)
            if not complete:
                status = ObservationStatus.MISSING
                reason = "DECLARED_HISTORY_WINDOW_UNAVAILABLE"
            elif not row.valid:
                status = ObservationStatus.INVALID
                reason = row.reason_code or "FEATURE_ROW_INVALID"
            elif needs_context and context_sha256[parent_index] is None:
                status = ObservationStatus.MISSING
                reason = "DONOR_CONTEXT_UNAVAILABLE"
            else:
                status = ObservationStatus.OBSERVED
                reason = "VALID"
            value: float | None
            if status is ObservationStatus.OBSERVED:
                value = _feature_value(spec, row)
                if not math.isfinite(value):
                    status = ObservationStatus.INVALID
                    reason = "NONFINITE_FEATURE_VALUE"
                    value = None
                    zero = None
                    opportunity = None
                    slots = None
                    censoring = CensoringStatus.NONE
                else:
                    zero = _physical_zero(value, spec)
                    opportunity, slots, censoring = _opportunity(spec, record)
            else:
                value = None
                zero = None
                opportunity = None
                slots = None
                censoring = CensoringStatus.NONE
            observation = CoordinateSupportObservation(
                parent_id=str(unit_ids[parent_index]),
                role_id=role_id,
                block_id=role_id,
                sector_id=int(k_values[parent_index]),
                fold_id=int(folds[parent_index]),
                feature_id=spec.feature_id,
                spec_sha256=spec.identity,
                observation_status=status,
                physical_value=value,
                physical_zero=zero,
                opportunity_mass=opportunity,
                opportunity_positive=(opportunity > 0.0 if opportunity is not None else None),
                clock_slot_count=slots,
                censoring_status=censoring,
                context_sha256=(context_sha256[parent_index] if needs_context else None),
                donor_set_sha256=(donor_set_sha256[parent_index] if needs_context else None),
                validity_reason=reason,
                source_prefix_sha256=prefix_sha,
            )
            observations.append(observation)
            by_feature[spec.feature_id].append(observation)
    weights = np.asarray(target_weights(list(k_values)), dtype=np.float64)
    ledger = tuple(
        _aggregate(
            spec,
            by_feature[spec.feature_id],
            k_values,
            weights,
            floors,
            context_passed=context_passed,
        )
        for spec in specs
    )
    disposition = {row.feature_id: row.disposition for row in ledger}
    acceptable = {
        CoordinateDisposition.RETAIN,
        CoordinateDisposition.RETAIN_CENSORED_WITH_ADEQUATE_EXPOSURE,
    }
    categorical_pass = all(disposition[name] in acceptable for name in CATEGORICAL_FEATURES)
    rc0_complete = all(
        disposition[name] in acceptable for name in COUNT_FEATURES if name != "empty_recent"
    ) and disposition["empty_recent"] in (
        acceptable | {CoordinateDisposition.PRUNE_ALLOWED_CONSTANT_TECHNICAL}
    )
    rc1_complete = rc0_complete and all(
        disposition[name] in acceptable for name in GEOMETRY_FEATURES
    )
    rc2_complete = rc1_complete and all(disposition[name] in acceptable for name in PAIR_FEATURES)
    retained_panels: list[Panel] = []
    if categorical_pass and rc0_complete:
        retained_panels.append(Panel.COUNT)
    if categorical_pass and rc1_complete:
        retained_panels.append(Panel.SPATIAL_RECENCY)
    if categorical_pass and rc2_complete:
        retained_panels.append(Panel.EVENT_PAIR)
    retained_numeric = tuple(name for name in ALL_FEATURES if disposition[name] in acceptable)
    rejected = {
        row.feature_id: row.disposition.value for row in ledger if row.disposition not in acceptable
    }
    admission = PanelAdmissionManifest(
        retained_panels=tuple(retained_panels),
        retained_features=retained_numeric,
        rejected_features=rejected,
        feature_order_by_panel={
            panel.value: feature_names(panel, retained_numeric) for panel in retained_panels
        },
        categorical_gate_passed=categorical_pass,
        group_complete={
            Panel.COUNT.value: rc0_complete,
            Panel.SPATIAL_RECENCY.value: rc1_complete,
            Panel.EVENT_PAIR.value: rc2_complete,
        },
    )
    return SupportAnalysis(
        specifications=specs,
        observations=tuple(observations),
        ledger_rows=ledger,
        admission=admission,
        support_spec_manifest_sha256=manifest_sha,
    )


__all__ = [
    "CensoringStatus",
    "CoordinateDisposition",
    "CoordinateSupportLedgerRow",
    "CoordinateSupportObservation",
    "CoordinateSupportSpec",
    "FeatureContextSupport",
    "ObservationStatus",
    "PanelAdmissionManifest",
    "PrefixSupportCommitment",
    "SupportAnalysis",
    "SupportClass",
    "SupportFloors",
    "analyze_support",
    "coordinate_support_specs",
    "evaluate_feature_context_support",
    "support_spec_manifest",
]
