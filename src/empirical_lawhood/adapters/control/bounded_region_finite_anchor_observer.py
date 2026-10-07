"""Outcome-blind observer for bounded regions plus exact finite anchors."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from fractions import Fraction
from math import gcd
from typing import ClassVar

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.atlases import ResponseAtlas
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.controller_study import ImplementationBinding, ImplementationRole
from empirical_lawhood.planning.evidence_geometry import ControlledMapAdmissionReceiptCorpus
from empirical_lawhood.planning.native_hold_decision_cell_calibration import NativeHoldCalibrationSelectedRoster, preparation_values_fingerprint
from empirical_lawhood.runtime.controller_runtime import (
    ObserverDisposition,
    ObserverEvaluation,
    RuntimeObservation,
)


MAX_BOUNDED_OBSERVER_REGIONS = 64
MAX_BOUNDED_OBSERVER_ANCHORS = 64
MAX_BOUNDED_OBSERVER_CONFIG_BYTES = 16 * 1024 * 1024


class DecimalQuantizationRule(StrEnum):
    ROUND_HALF_EVEN = "ROUND_HALF_EVEN"


@dataclass(frozen=True, slots=True)
class ObserverQuantitySpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/observer-quantity-spec'

    quantity_id: str
    native_unit: str

    def __post_init__(self) -> None:
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        validate_nonempty(self.native_unit, field_name="native_unit")


@dataclass(frozen=True, slots=True)
class DecimalInterval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/decimal-interval'

    interval_id: str
    lower: Decimal
    upper: Decimal
    lower_inclusive: bool
    upper_inclusive: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.interval_id, field_name="interval_id")
        validate_decimal(self.lower, field_name="lower")
        validate_decimal(self.upper, field_name="upper")
        if self.lower > self.upper or (
            self.lower == self.upper and not (self.lower_inclusive and self.upper_inclusive)
        ):
            raise ValueError("observer interval is empty or reversed")


@dataclass(frozen=True, slots=True)
class ExactRational(CanonicalRecord):
    """A normalized exact rational used where Decimal cannot encode 1/3."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/exact-rational'

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ValueError("exact rational denominator must be positive")
        common = gcd(abs(self.numerator), self.denominator)
        if common != 1:
            raise ValueError("exact rational must be normalized")


def _compare_rationals(left: ExactRational, right: ExactRational) -> int:
    difference = left.numerator * right.denominator - right.numerator * left.denominator
    return (difference > 0) - (difference < 0)


@dataclass(frozen=True, slots=True)
class ExactRationalInterval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/exact-rational-interval'

    interval_id: str
    lower: ExactRational
    upper: ExactRational
    lower_inclusive: bool
    upper_inclusive: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.interval_id, field_name="interval_id")
        comparison = _compare_rationals(self.lower, self.upper)
        if comparison > 0 or (
            comparison == 0 and not (self.lower_inclusive and self.upper_inclusive)
        ):
            raise ValueError("exact rational interval is empty or reversed")


@dataclass(frozen=True, slots=True)
class ObserverQuantityBound(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/observer-quantity-bound'

    bound_id: str
    quantity_id: str
    native_unit: str
    interval: DecimalInterval

    def __post_init__(self) -> None:
        validate_stable_id(self.bound_id, field_name="bound_id")
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        validate_nonempty(self.native_unit, field_name="native_unit")


@dataclass(frozen=True, slots=True)
class AffineDepthMap(CanonicalRecord):
    """Exact map ``quantity = intercept + slope * depth``."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/affine-depth-map'

    map_id: str
    quantity_id: str
    native_unit: str
    intercept: Decimal
    slope: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.map_id, field_name="map_id")
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_decimal(self.intercept, field_name="intercept")
        validate_decimal(self.slope, field_name="slope")
        if self.slope == 0:
            raise ValueError("affine observer depth map requires a nonzero slope")


@dataclass(frozen=True, slots=True)
class BoundedRegionDecisionCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/bounded-region-decision-cell'

    region_id: str
    decision_cell_id: str
    quantity_bounds: tuple[ObserverQuantityBound, ...]
    depth_map_ids: tuple[str, ...]
    depth_interval: ExactRationalInterval

    def __post_init__(self) -> None:
        validate_stable_id(self.region_id, field_name="region_id")
        validate_stable_id(self.decision_cell_id, field_name="decision_cell_id")
        require_sorted_unique_ids(
            self.quantity_bounds,
            attribute="quantity_id",
            field_name="quantity_bounds",
        )
        require_sorted_unique_strings(
            self.depth_map_ids,
            field_name="depth_map_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class FiniteAnchorDecisionCoordinate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/finite-anchor-decision-coordinate'

    anchor_id: str
    preparation_values: tuple[NamedDecimal, ...]
    preparation_fingerprint: str

    def __post_init__(self) -> None:
        validate_stable_id(self.anchor_id, field_name="anchor_id")
        require_sorted_unique_ids(
            self.preparation_values,
            attribute="value_id",
            field_name="preparation_values",
        )
        if not self.preparation_values:
            raise ValueError("finite observer anchor requires preparation values")
        validate_sha256(
            self.preparation_fingerprint,
            field_name="preparation_fingerprint",
        )
        if self.preparation_fingerprint != preparation_values_fingerprint(self.preparation_values):
            raise ValueError("finite observer anchor fingerprint differs from its values")


@dataclass(frozen=True, slots=True)
class _OpenClosedInterval:
    lower: Decimal
    upper: Decimal
    lower_inclusive: bool
    upper_inclusive: bool

    @classmethod
    def from_record(cls, value: DecimalInterval) -> _OpenClosedInterval:
        return cls(
            lower=value.lower,
            upper=value.upper,
            lower_inclusive=value.lower_inclusive,
            upper_inclusive=value.upper_inclusive,
        )

    @property
    def empty(self) -> bool:
        return self.lower > self.upper or (
            self.lower == self.upper and not (self.lower_inclusive and self.upper_inclusive)
        )


@dataclass(frozen=True, slots=True)
class _ExactOpenClosedInterval:
    lower: Fraction
    upper: Fraction
    lower_inclusive: bool
    upper_inclusive: bool

    @property
    def empty(self) -> bool:
        return self.lower > self.upper or (
            self.lower == self.upper and not (self.lower_inclusive and self.upper_inclusive)
        )


def _intersection(
    left: _OpenClosedInterval,
    right: _OpenClosedInterval,
) -> _OpenClosedInterval:
    lower = max(left.lower, right.lower)
    upper = min(left.upper, right.upper)
    lower_inclusive = (left.lower_inclusive if lower == left.lower else True) and (
        right.lower_inclusive if lower == right.lower else True
    )
    upper_inclusive = (left.upper_inclusive if upper == left.upper else True) and (
        right.upper_inclusive if upper == right.upper else True
    )
    return _OpenClosedInterval(lower, upper, lower_inclusive, upper_inclusive)


def _exact_intersection(
    left: _ExactOpenClosedInterval,
    right: _ExactOpenClosedInterval,
) -> _ExactOpenClosedInterval:
    lower = max(left.lower, right.lower)
    upper = min(left.upper, right.upper)
    lower_inclusive = (left.lower_inclusive if lower == left.lower else True) and (
        right.lower_inclusive if lower == right.lower else True
    )
    upper_inclusive = (left.upper_inclusive if upper == left.upper else True) and (
        right.upper_inclusive if upper == right.upper else True
    )
    return _ExactOpenClosedInterval(lower, upper, lower_inclusive, upper_inclusive)


def _decimal_intervals_overlap(left: DecimalInterval, right: DecimalInterval) -> bool:
    return not _intersection(
        _OpenClosedInterval.from_record(left),
        _OpenClosedInterval.from_record(right),
    ).empty


def _contains(interval: DecimalInterval, value: Decimal) -> bool:
    return (value > interval.lower or (value == interval.lower and interval.lower_inclusive)) and (
        value < interval.upper or (value == interval.upper and interval.upper_inclusive)
    )


def _compare_fraction_rational(value: Fraction, rational: ExactRational) -> int:
    difference = value.numerator * rational.denominator - rational.numerator * value.denominator
    return (difference > 0) - (difference < 0)


def _is_subset(
    inner: _ExactOpenClosedInterval,
    outer: ExactRationalInterval,
) -> bool:
    lower_comparison = _compare_fraction_rational(inner.lower, outer.lower)
    upper_comparison = _compare_fraction_rational(inner.upper, outer.upper)
    lower_ok = lower_comparison > 0 or (
        lower_comparison == 0 and (not inner.lower_inclusive or outer.lower_inclusive)
    )
    upper_ok = upper_comparison < 0 or (
        upper_comparison == 0 and (not inner.upper_inclusive or outer.upper_inclusive)
    )
    return not inner.empty and lower_ok and upper_ok


def _rational_intervals_overlap(
    left: ExactRationalInterval,
    right: ExactRationalInterval,
) -> bool:
    lower_comparison = _compare_rationals(left.lower, right.upper)
    if lower_comparison > 0 or (
        lower_comparison == 0 and not (left.lower_inclusive and right.upper_inclusive)
    ):
        return False
    upper_comparison = _compare_rationals(right.lower, left.upper)
    return not (
        upper_comparison > 0
        or (upper_comparison == 0 and not (right.lower_inclusive and left.upper_inclusive))
    )


def _exact_rational_intersects(
    left: _ExactOpenClosedInterval,
    right: ExactRationalInterval,
) -> bool:
    upper_to_lower = _compare_fraction_rational(left.upper, right.lower)
    if upper_to_lower < 0 or (
        upper_to_lower == 0 and not (left.upper_inclusive and right.lower_inclusive)
    ):
        return False
    lower_to_upper = _compare_fraction_rational(left.lower, right.upper)
    return not (
        lower_to_upper > 0
        or (lower_to_upper == 0 and not (left.lower_inclusive and right.upper_inclusive))
    )


def _quantized_preimage(value: Decimal, quantum: Decimal) -> _OpenClosedInterval | None:
    scaled = Fraction(value) / Fraction(quantum)
    if scaled.denominator != 1:
        return None
    tied_value_rounds_here = scaled.numerator % 2 == 0
    half = quantum / Decimal(2)
    return _OpenClosedInterval(
        value - half,
        value + half,
        tied_value_rounds_here,
        tied_value_rounds_here,
    )


def _inverse_affine_preimage(
    value: Decimal,
    mapping: AffineDepthMap,
    quantum: Decimal,
) -> _ExactOpenClosedInterval | None:
    preimage = _quantized_preimage(value, quantum)
    if preimage is None:
        return None
    intercept = Fraction(mapping.intercept)
    slope = Fraction(mapping.slope)
    first = (Fraction(preimage.lower) - intercept) / slope
    second = (Fraction(preimage.upper) - intercept) / slope
    if mapping.slope > 0:
        return _ExactOpenClosedInterval(
            first,
            second,
            preimage.lower_inclusive,
            preimage.upper_inclusive,
        )
    return _ExactOpenClosedInterval(
        second,
        first,
        preimage.upper_inclusive,
        preimage.lower_inclusive,
    )


def _boxes_overlap(
    left: BoundedRegionDecisionCell,
    right: BoundedRegionDecisionCell,
) -> bool:
    right_by_quantity = {value.quantity_id: value for value in right.quantity_bounds}
    return all(
        _decimal_intervals_overlap(
            bound.interval,
            right_by_quantity[bound.quantity_id].interval,
        )
        for bound in left.quantity_bounds
    )


@dataclass(frozen=True, slots=True)
class FiniteAnchorDecisionCoordinateTemplate(CanonicalRecord):
    """Preselection identity for one of the five exact native-HOLD slots."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/finite-anchor-decision-coordinate-template'

    anchor_id: str
    anchor_slot_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.anchor_id, field_name="anchor_id")
        validate_stable_id(self.anchor_slot_id, field_name="anchor_slot_id")


@dataclass(frozen=True, slots=True)
class BoundedRegionFiniteAnchorObserverTemplate(CanonicalRecord):
    "Observer design before measurement, without atlas, admission, or selected-anchor hashes."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/bounded-region-finite-anchor-observer-template'

    template_id: str
    expected_config_id: str
    expected_selected_anchor_roster_id: str
    expected_selected_anchor_roster_schema: str
    expected_atlas_id: str
    expected_atlas_schema: str
    expected_admission_corpus_id: str
    expected_admission_corpus_schema: str
    template_binder_implementation_id: str
    template_binder_implementation_sha256: str
    quantities: tuple[ObserverQuantitySpec, ...]
    decimal_quantum: Decimal
    quantization_rule: DecimalQuantizationRule
    affine_depth_maps: tuple[AffineDepthMap, ...]
    active_regions: tuple[BoundedRegionDecisionCell, ...]
    finite_anchor_slots: tuple[FiniteAnchorDecisionCoordinateTemplate, ...]
    hold_decision_cell_id: str
    classifier_identity: ObjectIdentity
    observer_reference: ExecutableReference
    observer_implementation_sha256: str
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for field_name, value in (
            ("template_id", self.template_id),
            ("expected_config_id", self.expected_config_id),
            (
                "expected_selected_anchor_roster_id",
                self.expected_selected_anchor_roster_id,
            ),
            ("expected_atlas_id", self.expected_atlas_id),
            ("expected_admission_corpus_id", self.expected_admission_corpus_id),
            (
                "template_binder_implementation_id",
                self.template_binder_implementation_id,
            ),
            ("hold_decision_cell_id", self.hold_decision_cell_id),
        ):
            validate_stable_id(value, field_name=field_name)
        for schema in (
            self.expected_selected_anchor_roster_schema,
            self.expected_atlas_schema,
            self.expected_admission_corpus_schema,
        ):
            validate_schema(schema)
        if (
            self.expected_selected_anchor_roster_schema
            != NativeHoldCalibrationSelectedRoster.SCHEMA
            or self.expected_atlas_schema != ResponseAtlas.SCHEMA
            or self.expected_admission_corpus_schema != ControlledMapAdmissionReceiptCorpus.SCHEMA
        ):
            raise ValueError("observer template expects another selected/admission schema")
        validate_sha256(
            self.template_binder_implementation_sha256,
            field_name="template_binder_implementation_sha256",
        )
        _validate_bounded_observer_static_design(
            quantities=self.quantities,
            decimal_quantum=self.decimal_quantum,
            quantization_rule=self.quantization_rule,
            affine_depth_maps=self.affine_depth_maps,
            active_regions=self.active_regions,
            finite_anchor_slots=self.finite_anchor_slots,
            hold_decision_cell_id=self.hold_decision_cell_id,
            observer_reference=self.observer_reference,
            observer_implementation_sha256=self.observer_implementation_sha256,
        )
        if (
            self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("bounded observer template must precede measurement and remain outcome-blind")


def _validate_bounded_observer_static_design(
    *,
    quantities: tuple[ObserverQuantitySpec, ...],
    decimal_quantum: Decimal,
    quantization_rule: DecimalQuantizationRule,
    affine_depth_maps: tuple[AffineDepthMap, ...],
    active_regions: tuple[BoundedRegionDecisionCell, ...],
    finite_anchor_slots: tuple[FiniteAnchorDecisionCoordinateTemplate, ...],
    hold_decision_cell_id: str,
    observer_reference: ExecutableReference,
    observer_implementation_sha256: str,
) -> None:
    require_sorted_unique_ids(quantities, attribute="quantity_id", field_name="quantities")
    if len(quantities) != 4:
        raise ValueError("bounded observer template requires exactly four quantities")
    validate_decimal(decimal_quantum, field_name="decimal_quantum", minimum=Decimal(0))
    if decimal_quantum == 0 or quantization_rule is not DecimalQuantizationRule.ROUND_HALF_EVEN:
        raise ValueError("bounded observer template changes exact decimal quantization")
    require_sorted_unique_ids(
        affine_depth_maps,
        attribute="map_id",
        field_name="affine_depth_maps",
    )
    if len(affine_depth_maps) != 2 or len({item.quantity_id for item in affine_depth_maps}) != 2:
        raise ValueError("bounded observer template requires exactly two affine maps")
    require_sorted_unique_ids(active_regions, attribute="region_id", field_name="active_regions")
    if len(active_regions) != 3:
        raise ValueError("bounded observer template requires exactly three active regions")
    require_sorted_unique_ids(
        finite_anchor_slots,
        attribute="anchor_id",
        field_name="finite_anchor_slots",
    )
    if (
        len(finite_anchor_slots) != 5
        or len({item.anchor_slot_id for item in finite_anchor_slots}) != 5
    ):
        raise ValueError("bounded observer template requires exactly five anchor slots")
    quantity_units = {item.quantity_id: item.native_unit for item in quantities}
    map_ids = {item.map_id for item in affine_depth_maps}
    if any(quantity_units.get(item.quantity_id) != item.native_unit for item in affine_depth_maps):
        raise ValueError("bounded observer template map changes quantity/unit")
    if len({item.decision_cell_id for item in active_regions}) != 3 or hold_decision_cell_id in {
        item.decision_cell_id for item in active_regions
    }:
        raise ValueError("bounded observer template aliases an active/HOLD cell")
    for region in active_regions:
        if {
            item.quantity_id: item.native_unit for item in region.quantity_bounds
        } != quantity_units or set(region.depth_map_ids) != map_ids:
            raise ValueError("bounded observer template changes region quantity/map coverage")
    for index, left in enumerate(active_regions):
        for right in active_regions[index + 1 :]:
            if _boxes_overlap(left, right) and _rational_intervals_overlap(
                left.depth_interval,
                right.depth_interval,
            ):
                raise ValueError("bounded observer template active regions overlap")
    if (
        observer_reference.input_schema != RuntimeObservation.SCHEMA
        or observer_reference.output_schema != ObserverEvaluation.SCHEMA
        or not observer_reference.deterministic
    ):
        raise ValueError("bounded observer template uses another observer port")
    validate_sha256(
        observer_implementation_sha256,
        field_name="observer_implementation_sha256",
    )


@dataclass(frozen=True, slots=True)
class BoundedRegionFiniteAnchorObserverConfig(CanonicalRecord):
    """Issued exact classifier configuration; it contains no outcome values."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/bounded-region-finite-anchor-observer-config'

    config_id: str
    quantities: tuple[ObserverQuantitySpec, ...]
    decimal_quantum: Decimal
    quantization_rule: DecimalQuantizationRule
    affine_depth_maps: tuple[AffineDepthMap, ...]
    active_regions: tuple[BoundedRegionDecisionCell, ...]
    finite_anchors: tuple[FiniteAnchorDecisionCoordinate, ...]
    hold_decision_cell_id: str
    selected_anchor_roster: NativeHoldCalibrationSelectedRoster
    atlas: ObjectIdentity
    admission_corpus: ObjectIdentity
    classifier_identity: ObjectIdentity
    observer_reference: ExecutableReference
    observer_implementation_sha256: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_ids(
            self.quantities,
            attribute="quantity_id",
            field_name="quantities",
        )
        if len(self.quantities) != 4:
            raise ValueError("bounded-region observer requires exactly four quantities")
        validate_decimal(
            self.decimal_quantum,
            field_name="decimal_quantum",
            minimum=Decimal(0),
        )
        if self.decimal_quantum == 0:
            raise ValueError("observer decimal quantum must be positive")
        if self.quantization_rule is not DecimalQuantizationRule.ROUND_HALF_EVEN:
            raise ValueError("observer quantization rule is not registered")
        require_sorted_unique_ids(
            self.affine_depth_maps,
            attribute="map_id",
            field_name="affine_depth_maps",
        )
        if (
            len(self.affine_depth_maps) != 2
            or len({value.quantity_id for value in self.affine_depth_maps}) != 2
        ):
            raise ValueError("bounded-region observer requires two distinct affine axes")
        require_sorted_unique_ids(
            self.active_regions,
            attribute="region_id",
            field_name="active_regions",
        )
        require_sorted_unique_ids(
            self.finite_anchors,
            attribute="anchor_id",
            field_name="finite_anchors",
        )
        if not self.active_regions or len(self.active_regions) > MAX_BOUNDED_OBSERVER_REGIONS:
            raise ValueError("active observer regions are empty or exceed their bound")
        if not self.finite_anchors or len(self.finite_anchors) > MAX_BOUNDED_OBSERVER_ANCHORS:
            raise ValueError("finite observer anchors are empty or exceed their bound")
        validate_stable_id(self.hold_decision_cell_id, field_name="hold_decision_cell_id")
        if len({value.decision_cell_id for value in self.active_regions}) != len(
            self.active_regions
        ):
            raise ValueError("active observer regions reuse a decision cell")
        if self.hold_decision_cell_id in {value.decision_cell_id for value in self.active_regions}:
            raise ValueError("finite HOLD cell aliases an active observer cell")
        quantity_units = {value.quantity_id: value.native_unit for value in self.quantities}
        map_ids = {value.map_id for value in self.affine_depth_maps}
        if any(
            quantity_units.get(value.quantity_id) != value.native_unit
            for value in self.affine_depth_maps
        ):
            raise ValueError("affine observer map uses an undeclared quantity or unit")
        for region in self.active_regions:
            if {
                value.quantity_id: value.native_unit for value in region.quantity_bounds
            } != quantity_units or set(region.depth_map_ids) != map_ids:
                raise ValueError("bounded observer region changes quantity/map coverage")
        for index, left in enumerate(self.active_regions):
            for right in self.active_regions[index + 1 :]:
                if _boxes_overlap(left, right) and _rational_intervals_overlap(
                    left.depth_interval,
                    right.depth_interval,
                ):
                    raise ValueError("bounded observer regions overlap")
        fingerprints = tuple(value.preparation_fingerprint for value in self.finite_anchors)
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("finite observer anchor appears more than once")
        for anchor in self.finite_anchors:
            if {
                value.value_id: value.unit for value in anchor.preparation_values
            } != quantity_units or any(
                _quantized_preimage(value.value, self.decimal_quantum) is None
                for value in anchor.preparation_values
            ):
                raise ValueError("finite observer anchor changes quantity, unit or quantization")
            matches, invalid = _matching_region_ids(self, anchor.preparation_values)
            if matches or invalid:
                raise ValueError("finite observer anchor lies in or ambiguously touches support")
        selected_anchors = tuple(
            sorted(
                (
                    value.selected.preparation_fingerprint,
                    value.selected.preparation_values,
                )
                for value in self.selected_anchor_roster.selections
            )
        )
        configured_anchors = tuple(
            sorted(
                (value.preparation_fingerprint, value.preparation_values)
                for value in self.finite_anchors
            )
        )
        if configured_anchors != selected_anchors:
            raise ValueError("observer anchors differ from the authenticated selected roster")
        if self.atlas.object_schema != 'empirical-lawhood/kernel/response-atlas':
            raise ValueError("observer config requires an exact response atlas")
        if self.admission_corpus.object_schema != 'empirical-lawhood/planning/controlled-map-admission-receipt-corpus':
            raise ValueError("observer config requires an exact raw-admission corpus")
        if (
            self.observer_reference.input_schema != RuntimeObservation.SCHEMA
            or self.observer_reference.output_schema != ObserverEvaluation.SCHEMA
            or not self.observer_reference.deterministic
        ):
            raise ValueError("observer executable reference has another port contract")
        validate_sha256(
            self.observer_implementation_sha256,
            field_name="observer_implementation_sha256",
        )
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("bounded-region observer must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class BoundedRegionFiniteAnchorObserverTemplateBindingReceipt(CanonicalRecord):
    "Identity receipt after admission for deterministic observer config materialization."

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/control/bounded-region-finite-anchor-observer-template-binding-receipt'
    )

    receipt_id: str
    template: ObjectIdentity
    selected_anchor_roster: ObjectIdentity
    atlas: ObjectIdentity
    admission_corpus: ObjectIdentity
    config: ObjectIdentity
    config_sha256: str
    binding_implementation_id: str
    binding_implementation_sha256: str
    outcome_dependent_choice: bool
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    scientific_verdict: None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(
            self.binding_implementation_id,
            field_name="binding_implementation_id",
        )
        validate_sha256(self.config_sha256, field_name="config_sha256")
        validate_sha256(
            self.binding_implementation_sha256,
            field_name="binding_implementation_sha256",
        )
        expected_schemas = (
            (self.template, BoundedRegionFiniteAnchorObserverTemplate.SCHEMA),
            (
                self.selected_anchor_roster,
                NativeHoldCalibrationSelectedRoster.SCHEMA,
            ),
            (self.atlas, ResponseAtlas.SCHEMA),
            (self.admission_corpus, ControlledMapAdmissionReceiptCorpus.SCHEMA),
            (self.config, BoundedRegionFiniteAnchorObserverConfig.SCHEMA),
        )
        if any(identity.object_schema != schema for identity, schema in expected_schemas):
            raise ValueError("bounded observer template binding contains another schema")
        if self.config.object_fingerprint != self.config_sha256:
            raise ValueError("bounded observer config SHA differs from canonical config identity")
        if self.outcome_dependent_choice or self.scientific_verdict is not None:
            raise ValueError("bounded observer template binding cannot select or adjudicate")
        if (
            self.evidence_ceiling is not EvidenceCeiling.ADMISSION
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("bounded observer binding must remain outcome-blind wiring after admission")


def bind_bounded_region_finite_anchor_observer_template(
    *,
    receipt_id: str,
    template: BoundedRegionFiniteAnchorObserverTemplate,
    selected_anchor_roster: NativeHoldCalibrationSelectedRoster,
    atlas: ResponseAtlas,
    admission_corpus: ControlledMapAdmissionReceiptCorpus,
    binding_implementation_id: str,
    binding_implementation_sha256: str,
) -> tuple[
    BoundedRegionFiniteAnchorObserverConfig,
    BoundedRegionFiniteAnchorObserverTemplateBindingReceipt,
]:
    "Attach selected anchors and exact admission identities without classifying a value."

    if (
        binding_implementation_id != template.template_binder_implementation_id
        or binding_implementation_sha256 != template.template_binder_implementation_sha256
    ):
        raise ValueError("bounded observer template binder changes implementation")
    roster_identity = ObjectIdentity.from_record(
        selected_anchor_roster.roster_id,
        selected_anchor_roster,
    )
    atlas_identity = ObjectIdentity.from_record(atlas.atlas_id, atlas)
    corpus_identity = ObjectIdentity.from_record(admission_corpus.corpus_id, admission_corpus)
    if (
        roster_identity.object_id != template.expected_selected_anchor_roster_id
        or roster_identity.object_schema != template.expected_selected_anchor_roster_schema
    ):
        raise ValueError("bounded observer binding substitutes selected anchor roster")
    if (
        atlas_identity.object_id != template.expected_atlas_id
        or atlas_identity.object_schema != template.expected_atlas_schema
    ):
        raise ValueError("bounded observer binding substitutes response atlas")
    if (
        corpus_identity.object_id != template.expected_admission_corpus_id
        or corpus_identity.object_schema != template.expected_admission_corpus_schema
    ):
        raise ValueError("bounded observer binding substitutes raw admission corpus")
    if ObjectIdentity.from_record(admission_corpus.plan.atlas.atlas_id, admission_corpus.plan.atlas) != (
        atlas_identity
    ):
        raise ValueError("bounded observer atlas and admission corpus differ")
    selected_by_slot = {item.slot_id: item for item in selected_anchor_roster.selections}
    if set(selected_by_slot) != {item.anchor_slot_id for item in template.finite_anchor_slots}:
        raise ValueError("bounded observer selected roster changes five anchor slots")
    finite_anchors = tuple(
        FiniteAnchorDecisionCoordinate(
            anchor_id=item.anchor_id,
            preparation_values=selected_by_slot[item.anchor_slot_id].selected.preparation_values,
            preparation_fingerprint=(
                selected_by_slot[item.anchor_slot_id].selected.preparation_fingerprint
            ),
        )
        for item in template.finite_anchor_slots
    )
    config = BoundedRegionFiniteAnchorObserverConfig(
        config_id=template.expected_config_id,
        quantities=template.quantities,
        decimal_quantum=template.decimal_quantum,
        quantization_rule=template.quantization_rule,
        affine_depth_maps=template.affine_depth_maps,
        active_regions=template.active_regions,
        finite_anchors=finite_anchors,
        hold_decision_cell_id=template.hold_decision_cell_id,
        selected_anchor_roster=selected_anchor_roster,
        atlas=atlas_identity,
        admission_corpus=corpus_identity,
        classifier_identity=template.classifier_identity,
        observer_reference=template.observer_reference,
        observer_implementation_sha256=template.observer_implementation_sha256,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    config_identity = ObjectIdentity.from_record(config.config_id, config)
    receipt = BoundedRegionFiniteAnchorObserverTemplateBindingReceipt(
        receipt_id=receipt_id,
        template=ObjectIdentity.from_record(template.template_id, template),
        selected_anchor_roster=roster_identity,
        atlas=atlas_identity,
        admission_corpus=corpus_identity,
        config=config_identity,
        config_sha256=config.fingerprint(),
        binding_implementation_id=binding_implementation_id,
        binding_implementation_sha256=binding_implementation_sha256,
        outcome_dependent_choice=False,
        evidence_ceiling=EvidenceCeiling.ADMISSION,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    return config, receipt


class BoundedRegionFiniteAnchorObserverTemplateBinder:
    "Code-owned deterministic observer-template binder after admission."

    def __init__(self, *, implementation_id: str, implementation_sha256: str) -> None:
        validate_stable_id(implementation_id, field_name="implementation_id")
        validate_sha256(implementation_sha256, field_name="implementation_sha256")
        self._implementation_id = implementation_id
        self._implementation_sha256 = implementation_sha256

    def bind(
        self,
        *,
        receipt_id: str,
        template: BoundedRegionFiniteAnchorObserverTemplate,
        selected_anchor_roster: NativeHoldCalibrationSelectedRoster,
        atlas: ResponseAtlas,
        admission_corpus: ControlledMapAdmissionReceiptCorpus,
    ) -> tuple[
        BoundedRegionFiniteAnchorObserverConfig,
        BoundedRegionFiniteAnchorObserverTemplateBindingReceipt,
    ]:
        return bind_bounded_region_finite_anchor_observer_template(
            receipt_id=receipt_id,
            template=template,
            selected_anchor_roster=selected_anchor_roster,
            atlas=atlas,
            admission_corpus=admission_corpus,
            binding_implementation_id=self._implementation_id,
            binding_implementation_sha256=self._implementation_sha256,
        )


def _matching_region_ids(
    config: BoundedRegionFiniteAnchorObserverConfig,
    values: tuple[NamedDecimal, ...],
) -> tuple[tuple[str, ...], bool]:
    by_quantity = {value.value_id: value for value in values}
    maps = {value.map_id: value for value in config.affine_depth_maps}
    box_regions = tuple(
        region
        for region in config.active_regions
        if all(
            bound.native_unit == by_quantity[bound.quantity_id].unit
            and _contains(bound.interval, by_quantity[bound.quantity_id].value)
            for bound in region.quantity_bounds
        )
    )
    if not box_regions:
        return (), False
    candidates: list[str] = []
    invalid_inverse = False
    depth_preimages: list[tuple[BoundedRegionDecisionCell, _ExactOpenClosedInterval]] = []
    for region in box_regions:
        depth: _ExactOpenClosedInterval | None = None
        for map_id in region.depth_map_ids:
            mapping = maps[map_id]
            inverse = _inverse_affine_preimage(
                by_quantity[mapping.quantity_id].value,
                mapping,
                config.decimal_quantum,
            )
            if inverse is None:
                invalid_inverse = True
                depth = None
                break
            depth = inverse if depth is None else _exact_intersection(depth, inverse)
        if depth is None:
            invalid_inverse = True
            continue
        if depth.empty:
            continue
        depth_preimages.append((region, depth))
        if _is_subset(depth, region.depth_interval):
            candidates.append(region.region_id)
    if len(candidates) == 1 and not invalid_inverse:
        return tuple(candidates), False
    if len(candidates) > 1:
        return tuple(sorted(candidates)), True
    ambiguous_boundary = any(
        _exact_rational_intersects(depth, region.depth_interval)
        and not _is_subset(depth, region.depth_interval)
        for region, depth in depth_preimages
    )
    crosses_tiers = any(
        sum(_exact_rational_intersects(depth, region.depth_interval) for region in box_regions) > 1
        for _, depth in depth_preimages
    )
    return (), invalid_inverse or ambiguous_boundary or crosses_tiers


class BoundedRegionFiniteAnchorObserver:
    """Classify one exact active region or one exact finite HOLD anchor."""

    def __init__(
        self,
        *,
        config: BoundedRegionFiniteAnchorObserverConfig,
        implementation_binding: ImplementationBinding,
        config_artifact: ArtifactIdentity,
    ) -> None:
        if implementation_binding.role is not ImplementationRole.OBSERVER:
            raise ValueError("bounded-region observer requires OBSERVER role")
        if (
            implementation_binding.reference != config.observer_reference
            or implementation_binding.implementation_sha256 != config.observer_implementation_sha256
            or implementation_binding.config_sha256 != config.fingerprint()
        ):
            raise ValueError("bounded-region observer implementation/config binding differs")
        config_bytes = config.canonical_bytes()
        if (
            config_artifact.payload_schema != config.SCHEMA
            or config_artifact.sha256 != config.fingerprint()
            or config_artifact.size_bytes != len(config_bytes)
            or config_artifact.media_type != "application/json"
        ):
            raise ValueError("bounded-region observer config artifact differs from config bytes")
        self.config = config
        self.implementation_binding = implementation_binding
        self.config_artifact = config_artifact

    def _evaluation(
        self,
        observation: RuntimeObservation,
        *,
        disposition: ObserverDisposition,
        decision_cell_id: str | None = None,
        candidate_cell_ids: tuple[str, ...] = (),
        reason_codes: tuple[str, ...] = (),
    ) -> ObserverEvaluation:
        return ObserverEvaluation(
            evaluation_id=(
                f"observer-evaluation.{self.config.config_id}.{observation.observation_id}"
            ),
            observation=ObjectIdentity.from_record(observation.observation_id, observation),
            observer=self.implementation_binding,
            disposition=disposition,
            resolved_decision_cell_id=decision_cell_id,
            candidate_cell_ids=candidate_cell_ids,
            receiver_candidate_pair_ids=(),
            receiver_certificate=None,
            reason_codes=reason_codes,
        )

    def observe(self, observation: RuntimeObservation) -> ObserverEvaluation:
        expected_units = {value.quantity_id: value.native_unit for value in self.config.quantities}
        values = {value.value_id: value for value in observation.values}
        if self.config_artifact not in observation.input_artifacts:
            return self._evaluation(
                observation,
                disposition=ObserverDisposition.UNEVALUABLE,
                reason_codes=("OBSERVER_CONFIG_EVIDENCE_MISSING",),
            )
        if any(
            quantity_id not in values
            or values[quantity_id].unit != native_unit
            or _quantized_preimage(
                values[quantity_id].value,
                self.config.decimal_quantum,
            )
            is None
            for quantity_id, native_unit in expected_units.items()
        ):
            return self._evaluation(
                observation,
                disposition=ObserverDisposition.UNEVALUABLE,
                reason_codes=("OBSERVER_CLASSIFIER_OBSERVATION_INVALID",),
            )
        preparation_values = tuple(values[key] for key in sorted(expected_units))
        preparation_fingerprint = preparation_values_fingerprint(preparation_values)
        anchors = tuple(
            value
            for value in self.config.finite_anchors
            if value.preparation_fingerprint == preparation_fingerprint
            and value.preparation_values == preparation_values
        )
        region_ids, invalid = _matching_region_ids(self.config, preparation_values)
        if len(anchors) == 1 and not region_ids and not invalid:
            cell_id = self.config.hold_decision_cell_id
            return self._evaluation(
                observation,
                disposition=ObserverDisposition.EXACT,
                decision_cell_id=cell_id,
                candidate_cell_ids=(cell_id,),
            )
        if len(region_ids) == 1 and not anchors and not invalid:
            region = next(
                value for value in self.config.active_regions if value.region_id == region_ids[0]
            )
            return self._evaluation(
                observation,
                disposition=ObserverDisposition.EXACT,
                decision_cell_id=region.decision_cell_id,
                candidate_cell_ids=(region.decision_cell_id,),
            )
        if anchors or region_ids or invalid:
            return self._evaluation(
                observation,
                disposition=ObserverDisposition.UNEVALUABLE,
                reason_codes=("OBSERVER_CLASSIFICATION_AMBIGUOUS_OR_INVALID",),
            )
        return self._evaluation(
            observation,
            disposition=ObserverDisposition.OUTSIDE_SUPPORT,
            reason_codes=("BOUNDED_REGION_FINITE_ANCHOR_OUTSIDE_SUPPORT",),
        )


def decode_bounded_region_finite_anchor_observer_template(
    payload: bytes,
) -> BoundedRegionFiniteAnchorObserverTemplate:
    return decode_canonical_bytes(
        payload,
        BoundedRegionFiniteAnchorObserverTemplate,
        maximum_bytes=MAX_BOUNDED_OBSERVER_CONFIG_BYTES,
    )


def decode_bounded_region_finite_anchor_observer_template_binding_receipt(
    payload: bytes,
) -> BoundedRegionFiniteAnchorObserverTemplateBindingReceipt:
    return decode_canonical_bytes(
        payload,
        BoundedRegionFiniteAnchorObserverTemplateBindingReceipt,
        maximum_bytes=MAX_BOUNDED_OBSERVER_CONFIG_BYTES,
    )


__all__ = [
    'AffineDepthMap',
    'BoundedRegionDecisionCell',
    'BoundedRegionFiniteAnchorObserverConfig',
    'BoundedRegionFiniteAnchorObserverTemplateBinder',
    'BoundedRegionFiniteAnchorObserverTemplateBindingReceipt',
    'BoundedRegionFiniteAnchorObserverTemplate',
    'BoundedRegionFiniteAnchorObserver',
    'DecimalInterval',
    'DecimalQuantizationRule',
    'ExactRationalInterval',
    'ExactRational',
    'FiniteAnchorDecisionCoordinate',
    'FiniteAnchorDecisionCoordinateTemplate',
    "MAX_BOUNDED_OBSERVER_CONFIG_BYTES",
    "MAX_BOUNDED_OBSERVER_ANCHORS",
    "MAX_BOUNDED_OBSERVER_REGIONS",
    'ObserverQuantityBound',
    'ObserverQuantitySpec',
    'bind_bounded_region_finite_anchor_observer_template',
    'decode_bounded_region_finite_anchor_observer_template_binding_receipt',
    'decode_bounded_region_finite_anchor_observer_template',
]
