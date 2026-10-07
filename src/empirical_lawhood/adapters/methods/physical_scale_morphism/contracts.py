"""Strict, apparatus-neutral contracts for the physical scale morphism method package.

These records describe finite statistical experiments and property-indexed
morphisms.  They deliberately do not contain RC-ladder source bytes or an
apparatus-specific decoder; those belong to the simulator and physical
packages.  The contracts remain package-local because no second current
experiment consumes this exact morphism grammar.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)


PHYSICAL_SCALE_MORPHISM_COORDINATE_IDS = ("A", "D", "H", "R", "tau")
PHYSICAL_SCALE_MORPHISM_FORECAST_LEVEL_IDS = ("categorical", "dynamical", "metric")
PHYSICAL_SCALE_MORPHISM_SEMANTIC_ROLE_IDS = (
    "action",
    "denominator",
    "effort",
    "history",
    "hold",
    "preservation",
    "receiver",
    "sink",
    "target",
    "validity",
)


class PhysicalScaleMorphismEvidenceWorld(StrEnum):
    ANALYTIC_TRUTH = "ANALYTIC_TRUTH"
    NUMERICAL_TWIN = "NUMERICAL_TWIN"
    PHYSICAL_RC = "PHYSICAL_RC"


class PhysicalScaleMorphismMapFamily(StrEnum):
    RECEIVER = "RECEIVER"
    NUMERICAL = "NUMERICAL"
    PHYSICAL_SCALE = "PHYSICAL_SCALE"
    COMPOSITION = "COMPOSITION"


class PhysicalScaleMorphismLossiness(StrEnum):
    LOSSLESS = "LOSSLESS"
    LOSSY = "LOSSY"
    SET_VALUED = "SET_VALUED"


class PhysicalScaleMorphismForecastLevel(StrEnum):
    CATEGORICAL = "categorical"
    DYNAMICAL = "dynamical"
    METRIC = "metric"


class PhysicalScaleMorphismForecastState(StrEnum):
    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    UNEVALUABLE = "UNEVALUABLE"


class PhysicalScaleMorphismGateSign(StrEnum):
    AMBIGUOUS = "AMBIGUOUS"
    FAIL = "FAIL"
    OUT_OF_SUPPORT = "OUT_OF_SUPPORT"
    PASS = "PASS"
    UNEVALUABLE = "UNEVALUABLE"


class PhysicalScaleMorphismHoldDisposition(StrEnum):
    HOLD_INVALID_OR_NOT_REALIZED = "HOLD_INVALID_OR_NOT_REALIZED"
    HOLD_MEASURED_SAFE = "HOLD_MEASURED_SAFE"
    HOLD_MEASURED_UNSAFE = "HOLD_MEASURED_UNSAFE"
    HOLD_UNQUALIFIED = "HOLD_UNQUALIFIED"


class PhysicalScaleMorphismObstructionKind(StrEnum):
    ABSENT_OPERAND = "ABSENT_OPERAND"
    ACTION_CHAIN = "ACTION_CHAIN"
    AUTHORITY = "AUTHORITY"
    NONATTEMPT = "NONATTEMPT"
    OBSERVED_OPPOSITION = "OBSERVED_OPPOSITION"
    POWER = "POWER"
    SUPPORT_OR_DENOMINATOR = "SUPPORT_OR_DENOMINATOR"
    UNRESOLVED = "UNRESOLVED"


class PhysicalScaleMorphismComparatorKind(StrEnum):
    ACTION_ONLY = "ACTION_ONLY"
    D_A = "D_A"
    FULL_TUPLE = "FULL_TUPLE"
    H_A = "H_A"
    R_A = "R_A"
    SATURATED_DEVELOPMENT_LOOKUP = "SATURATED_DEVELOPMENT_LOOKUP"
    SELECTED_TYPED_SUBSET = "SELECTED_TYPED_SUBSET"
    TARGET_NATIVE_LABEL = "TARGET_NATIVE_LABEL"
    TAU_A = "TAU_A"
    WILDCARD = "WILDCARD"


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismStatisticalExperiment(CanonicalRecord):
    """Identity of one finite experiment without embedding its observations."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-statistical-experiment'

    experiment_id: str
    evidence_world: PhysicalScaleMorphismEvidenceWorld
    implementation_id: str
    denominator_id: str
    physical_scale_cells: int
    numerical_view_id: str
    receiver_id: str
    history_id: str
    preparation_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    horizon_ids: tuple[str, ...]
    gate_operand_ids: tuple[str, ...]
    independent_unit: str
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "experiment_id",
            "implementation_id",
            "denominator_id",
            "numerical_view_id",
            "receiver_id",
            "history_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.physical_scale_cells not in {1, 2, 4, 8, 16, 32, 64}:
            raise ValueError("physical scale morphism scale lies outside the finite method/apparatus roster")
        for name in (
            "preparation_ids",
            "action_ids",
            "horizon_ids",
            "gate_operand_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        validate_nonempty(self.independent_unit, field_name="independent_unit")
        if self.evidence_world is PhysicalScaleMorphismEvidenceWorld.PHYSICAL_RC:
            if self.numerical_view_id != "physical":
                raise ValueError("physical experiment cannot carry a numerical solver view")
            if self.independent_unit != "independently-identified-board":
                raise ValueError("physical physical scale morphism inference must retain board as its unit")
        elif self.numerical_view_id == "physical":
            raise ValueError("nonphysical experiment cannot use the physical view identity")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismPropertyPrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-property-prediction'

    prediction_id: str
    property_id: str
    forecast_level: PhysicalScaleMorphismForecastLevel
    predicted_to_survive: bool
    support_ids: tuple[str, ...]
    falsifier_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.prediction_id, field_name="prediction_id")
        validate_stable_id(self.property_id, field_name="property_id")
        require_sorted_unique_strings(self.support_ids, field_name="support_ids", allow_empty=False)
        require_sorted_unique_strings(
            self.falsifier_ids, field_name="falsifier_ids", allow_empty=False
        )


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismMorphismRecord(CanonicalRecord):
    """One frozen, property-indexed map between statistical experiments."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-morphism-record'

    morphism_id: str
    source_experiment: ObjectIdentity
    target_experiment: ObjectIdentity
    map_family: PhysicalScaleMorphismMapFamily
    direction: str
    component_map_ids: tuple[str, ...]
    native_unit_transform_ids: tuple[str, ...]
    dimensionless_transform_ids: tuple[str, ...]
    support_ids: tuple[str, ...]
    excluded_boundary_ids: tuple[str, ...]
    deterministic: bool
    lossiness: PhysicalScaleMorphismLossiness
    invertible: bool
    discarded_information_ids: tuple[str, ...]
    action_correspondence_id: str
    causal_cutoff_id: str
    receiver_window_id: str
    complete_board_linkage_required: bool
    preserved_role_ids: tuple[str, ...]
    omitted_role_ids: tuple[str, ...]
    merged_role_ids: tuple[str, ...]
    predictions: tuple[PhysicalScaleMorphismPropertyPrediction, ...]
    direct_map_id: str | None
    composed_map_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.morphism_id, field_name="morphism_id")
        if self.source_experiment.object_schema != PhysicalScaleMorphismStatisticalExperiment.SCHEMA:
            raise ValueError("morphism source is not an physical scale morphism statistical experiment")
        if self.target_experiment.object_schema != PhysicalScaleMorphismStatisticalExperiment.SCHEMA:
            raise ValueError("morphism target is not an physical scale morphism statistical experiment")
        validate_nonempty(self.direction, field_name="direction")
        for name in (
            "component_map_ids",
            "native_unit_transform_ids",
            "dimensionless_transform_ids",
            "support_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        for name in (
            "excluded_boundary_ids",
            "discarded_information_ids",
            "preserved_role_ids",
            "omitted_role_ids",
            "merged_role_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        for index, value in enumerate(self.composed_map_ids):
            validate_stable_id(value, field_name=f"composed_map_ids[{index}]")
        if len(set(self.composed_map_ids)) != len(self.composed_map_ids):
            raise ValueError("composed_map_ids must preserve an ordered unique map chain")
        for name in (
            "action_correspondence_id",
            "causal_cutoff_id",
            "receiver_window_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if not self.complete_board_linkage_required:
            raise ValueError("physical scale morphism morphisms require complete board/context linkage")
        role_sets = (
            set(self.preserved_role_ids),
            set(self.omitted_role_ids),
            set(self.merged_role_ids),
        )
        if any(
            left & right for index, left in enumerate(role_sets) for right in role_sets[index + 1 :]
        ):
            raise ValueError("semantic role dispositions must be disjoint")
        if not set().union(*role_sets).issubset(PHYSICAL_SCALE_MORPHISM_SEMANTIC_ROLE_IDS):
            raise ValueError("morphism contains an unknown semantic role")
        require_sorted_unique_ids(
            self.predictions, attribute="prediction_id", field_name="predictions"
        )
        if len({(value.property_id, value.forecast_level) for value in self.predictions}) != len(
            self.predictions
        ):
            raise ValueError("morphism property/forecast predictions repeat")
        if self.lossiness is PhysicalScaleMorphismLossiness.LOSSLESS and self.discarded_information_ids:
            raise ValueError("lossless morphism cannot declare discarded information")
        if self.lossiness is not PhysicalScaleMorphismLossiness.LOSSLESS and not self.discarded_information_ids:
            raise ValueError("lossy/set-valued morphism must state discarded information")
        if self.invertible and self.lossiness is not PhysicalScaleMorphismLossiness.LOSSLESS:
            raise ValueError("lossy physical scale morphism morphism cannot be declared invertible")
        if self.direct_map_id is not None:
            validate_stable_id(self.direct_map_id, field_name="direct_map_id")
        if self.composed_map_ids and self.direct_map_id is None:
            raise ValueError(
                "composition comparison requires an independently identified direct map"
            )


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismForecastMember(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-forecast-member'

    member_id: str
    level: PhysicalScaleMorphismForecastLevel
    property_ids: tuple[str, ...]
    coordinate_ids: tuple[str, ...]
    prediction_artifact_id: str
    scoring_rule_id: str
    tolerance: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        require_sorted_unique_strings(
            self.property_ids, field_name="property_ids", allow_empty=False
        )
        require_sorted_unique_strings(
            self.coordinate_ids, field_name="coordinate_ids", allow_empty=False
        )
        validate_stable_id(self.prediction_artifact_id, field_name="prediction_artifact_id")
        validate_stable_id(self.scoring_rule_id, field_name="scoring_rule_id")
        validate_decimal(self.tolerance, field_name="tolerance", minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismForecastTriplet(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-forecast-triplet'

    triplet_id: str
    complete_unit_ids: tuple[str, ...]
    assignment_id: str
    causal_cutoff_id: str
    receiver_window_id: str
    horizon_id: str
    members: tuple[PhysicalScaleMorphismForecastMember, ...]
    multiplicity_family_id: str
    frozen_before_evaluation: bool
    protected_evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "triplet_id",
            "assignment_id",
            "causal_cutoff_id",
            "receiver_window_id",
            "horizon_id",
            "multiplicity_family_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.complete_unit_ids, field_name="complete_unit_ids", allow_empty=False
        )
        require_sorted_unique_ids(self.members, attribute="member_id", field_name="members")
        levels = tuple(sorted(value.level.value for value in self.members))
        if levels != PHYSICAL_SCALE_MORPHISM_FORECAST_LEVEL_IDS:
            raise ValueError("forecast triplet requires categorical, dynamical and metric members")
        if not self.frozen_before_evaluation or self.protected_evaluation_outcome_count:
            raise ValueError("forecast triplet crossed its evaluation freeze")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("forecast triplet must be authored outcome blind")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismForecastMemberScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-forecast-member-score'

    score_id: str
    level: PhysicalScaleMorphismForecastLevel
    state: PhysicalScaleMorphismForecastState
    complete_unit_count: int
    mismatch_unit_ids: tuple[str, ...]
    maximum_defect: Decimal | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.score_id, field_name="score_id")
        if self.complete_unit_count < 0:
            raise ValueError("forecast complete-unit count must be nonnegative")
        require_sorted_unique_strings(self.mismatch_unit_ids, field_name="mismatch_unit_ids")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.maximum_defect is not None:
            validate_decimal(self.maximum_defect, field_name="maximum_defect", minimum=Decimal(0))
        if self.state is PhysicalScaleMorphismForecastState.SUPPORTED and (
            self.mismatch_unit_ids or self.reason_codes
        ):
            raise ValueError("supported forecast member cannot retain mismatches")
        if self.state is not PhysicalScaleMorphismForecastState.SUPPORTED and not self.reason_codes:
            raise ValueError("non-supported forecast member requires reason codes")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismForecastTripletScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-forecast-triplet-score'

    score_id: str
    triplet: ObjectIdentity
    member_scores: tuple[PhysicalScaleMorphismForecastMemberScore, ...]
    paired_triplet_supported: bool
    categorical_only: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.score_id, field_name="score_id")
        if self.triplet.object_schema != PhysicalScaleMorphismForecastTriplet.SCHEMA:
            raise ValueError("triplet score binds the wrong record schema")
        require_sorted_unique_ids(
            self.member_scores, attribute="score_id", field_name="member_scores"
        )
        levels = {value.level: value.state for value in self.member_scores}
        if set(levels) != set(PhysicalScaleMorphismForecastLevel):
            raise ValueError("triplet score must cover all three forecast levels")
        expected_supported = all(value is PhysicalScaleMorphismForecastState.SUPPORTED for value in levels.values())
        expected_categorical_only = (
            levels[PhysicalScaleMorphismForecastLevel.CATEGORICAL] is PhysicalScaleMorphismForecastState.SUPPORTED
            and not expected_supported
        )
        if self.paired_triplet_supported != expected_supported:
            raise ValueError("paired-triplet support is not derived from member scores")
        if self.categorical_only != expected_categorical_only:
            raise ValueError("categorical-only state is not derived from member scores")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismSemanticRolePanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-semantic-role-panel'

    panel_id: str
    morphism_id: str
    preserved_role_ids: tuple[str, ...]
    omitted_role_ids: tuple[str, ...]
    merged_role_ids: tuple[str, ...]
    present_close_pair_ids: tuple[str, ...]
    future_diverged_pair_ids: tuple[str, ...]
    closure_supported: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        validate_stable_id(self.morphism_id, field_name="morphism_id")
        for name in (
            "preserved_role_ids",
            "omitted_role_ids",
            "merged_role_ids",
            "present_close_pair_ids",
            "future_diverged_pair_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if not set(self.future_diverged_pair_ids).issubset(self.present_close_pair_ids):
            raise ValueError("future-diverged fibres must first be present-close")
        expected = not (
            self.omitted_role_ids or self.merged_role_ids or self.future_diverged_pair_ids
        )
        if self.closure_supported != expected:
            raise ValueError("semantic/future closure is not data-derived")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismHoldFibreRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-hold-fibre-record'

    hold_id: str
    context_id: str
    board_id: str
    realized_episode_id: str | None
    gate_signs: tuple[PhysicalScaleMorphismGateSign, ...]
    disposition: PhysicalScaleMorphismHoldDisposition
    mapped_disposition: PhysicalScaleMorphismHoldDisposition | None
    false_safe_hold: bool

    def __post_init__(self) -> None:
        for name in ("hold_id", "context_id", "board_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.realized_episode_id is not None:
            validate_stable_id(self.realized_episode_id, field_name="realized_episode_id")
        if not self.gate_signs:
            raise ValueError("hold fibre requires its gate vector")
        if self.realized_episode_id is None:
            if self.disposition is not PhysicalScaleMorphismHoldDisposition.HOLD_INVALID_OR_NOT_REALIZED:
                raise ValueError("unrealized hold must retain its exact invalid disposition")
        elif self.disposition is PhysicalScaleMorphismHoldDisposition.HOLD_INVALID_OR_NOT_REALIZED:
            raise ValueError("realized hold cannot be marked not realized")
        if self.disposition is PhysicalScaleMorphismHoldDisposition.HOLD_MEASURED_SAFE:
            if any(value is not PhysicalScaleMorphismGateSign.PASS for value in self.gate_signs):
                raise ValueError("safe hold requires every assessed gate to pass")
        expected_false_safe = (
            self.mapped_disposition is PhysicalScaleMorphismHoldDisposition.HOLD_MEASURED_SAFE
            and (self.disposition is not PhysicalScaleMorphismHoldDisposition.HOLD_MEASURED_SAFE)
        )
        if self.false_safe_hold != expected_false_safe:
            raise ValueError("false-safe hold is not derived from fine and mapped dispositions")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismDefectPanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-defect-panel'

    panel_id: str
    morphism_id: str
    calibration_defect: Decimal | None
    observational_defect: Decimal | None
    held_future_semantic_defect: Decimal | None
    interventional_defect: Decimal | None
    decision_defect: Decimal | None
    unmatched_operand_ids: tuple[str, ...]
    calibration_only: bool
    observational_only: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        validate_stable_id(self.morphism_id, field_name="morphism_id")
        values = (
            self.calibration_defect,
            self.observational_defect,
            self.held_future_semantic_defect,
            self.interventional_defect,
            self.decision_defect,
        )
        for index, value in enumerate(values):
            if value is not None:
                validate_decimal(value, field_name=f"defect[{index}]", minimum=Decimal(0))
        require_sorted_unique_strings(
            self.unmatched_operand_ids, field_name="unmatched_operand_ids"
        )
        if all(value is None for value in values):
            raise ValueError("defect panel cannot be empty")
        if self.calibration_only and self.calibration_defect is None:
            raise ValueError("calibration-only panel requires calibration evidence")
        if self.observational_only and self.observational_defect is None:
            raise ValueError("observational-only panel requires observational evidence")
        if self.calibration_only and self.observational_only:
            raise ValueError("calibration-only and observational-only are distinct terminals")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismObstructionSet(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-obstruction-set'

    obstruction_id: str
    owner_id: str
    kinds: tuple[PhysicalScaleMorphismObstructionKind, ...]
    reason_codes: tuple[str, ...]
    permitted_next_act_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.obstruction_id, field_name="obstruction_id")
        validate_stable_id(self.owner_id, field_name="owner_id")
        if tuple(sorted(set(self.kinds), key=lambda value: value.value)) != self.kinds:
            raise ValueError("obstruction kinds must be sorted and unique")
        require_sorted_unique_strings(
            self.reason_codes, field_name="reason_codes", allow_empty=not self.kinds
        )
        require_sorted_unique_strings(
            self.permitted_next_act_ids,
            field_name="permitted_next_act_ids",
            allow_empty=not self.kinds,
        )
        if bool(self.kinds) != bool(self.reason_codes):
            raise ValueError("obstruction reasons and kinds must be jointly present")


__all__ = [
    'PhysicalScaleMorphismComparatorKind',
    "PHYSICAL_SCALE_MORPHISM_COORDINATE_IDS",
    'PhysicalScaleMorphismDefectPanel',
    'PhysicalScaleMorphismEvidenceWorld',
    'PhysicalScaleMorphismForecastLevel',
    'PhysicalScaleMorphismForecastMember',
    'PhysicalScaleMorphismForecastMemberScore',
    'PhysicalScaleMorphismForecastState',
    'PhysicalScaleMorphismForecastTriplet',
    'PhysicalScaleMorphismForecastTripletScore',
    'PhysicalScaleMorphismGateSign',
    'PhysicalScaleMorphismHoldDisposition',
    'PhysicalScaleMorphismHoldFibreRecord',
    'PhysicalScaleMorphismLossiness',
    'PhysicalScaleMorphismMapFamily',
    'PhysicalScaleMorphismMorphismRecord',
    'PhysicalScaleMorphismObstructionKind',
    'PhysicalScaleMorphismObstructionSet',
    'PhysicalScaleMorphismPropertyPrediction',
    'PhysicalScaleMorphismSemanticRolePanel',
    'PhysicalScaleMorphismStatisticalExperiment',
]
