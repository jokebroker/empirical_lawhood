"Outcome-blind authoring for matched action-versus-qualified-HOLD controller use.\n\nThe matched design keeps one physical efficacy preparation as the inferential\nunit.  Numerical members and the committed-action/qualified-HOLD executions\nremain nested evidence.  Outside-support HOLD controls are a separate roster\nand never enter the efficacy statistic.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ActionWordSupportStatus
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
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
from empirical_lawhood.kernel.time import ClockCoordinate
from empirical_lawhood.planning.controller_study import AdmissionControllerStudy, EvaluationStratumSpec, EvaluatorBoundarySpec, ImplementationBinding, ImplementationRole, AdmissionMeasuredHoldFibre, UtilityDirection
from empirical_lawhood.planning.finite_chart_reference import FiniteChartReferenceDesign
from empirical_lawhood.planning.native_hold_decision_cell_calibration import NativeHoldCalibrationDisposition, NativeHoldCalibrationSelectedRoster, NativeHoldDecisionCellCalibrationReceipt


MAX_MATCHED_ACTION_HOLD_PLAN_BYTES = 16 * 1024 * 1024

MATCHED_ACTION_HOLD_REDUCTION_ORDER = (
    "ACTION_MINUS_QUALIFIED_HOLD_WITHIN_MEMBER",
    "MEMBERWISE_MINIMUM_WITHIN_PHYSICAL_UNIT",
    "PHYSICAL_INDEPENDENT_UNIT_STUDENT_INFERENCE",
)

MATCHED_ACTION_HOLD_TERMINAL_PRECEDENCE = (
    "AUTHORITY_OR_RESOURCE_STOP",
    "RECEIPT_CUSTODY_OR_REVEAL_INVALID",
    "UNSAFE_OR_FALSE_SAFE_DISPOSITION",
    "DELIVERY_INVALID",
    "TECHNICAL_PARTIAL",
    "EFFICACY_UNEVALUABLE",
    "MIXED_MEMBER_OR_STRATUM_EFFICACY",
    "VALID_NEGATIVE_OR_SUBMATERIAL_EFFICACY",
    "CONTROLLER_USE_HOLD_CONTROL_FAILED",
    "TERMINAL_PRIMARY_CONTROLLER_USE_REPLICATION_POSITIVE",
)

MATCHED_ACTION_HOLD_SEALED_BUNDLE_SCHEMA = (
    'empirical-lawhood/runtime/matched-action-hold-prospective-bundle'
)
MATCHED_ACTION_HOLD_REVEALED_BUNDLE_SCHEMA = (
    'empirical-lawhood/runtime/matched-action-hold-revealed-bundle'
)
MATCHED_ACTION_HOLD_ADJUDICATION_SCHEMA = (
    'empirical-lawhood/runtime/matched-action-hold-cohort-adjudication'
)
MATCHED_ACTION_HOLD_EVALUATION_TEMPLATE_SCHEMA = (
    'empirical-lawhood/planning/matched-action-hold-controller-evaluation-template'
)
MATCHED_ACTION_HOLD_SELECTED_PREPARATION_ROSTER_SCHEMA = (
    'empirical-lawhood/planning/matched-action-hold-selected-preparation-roster'
)
MATCHED_ACTION_HOLD_OBSERVER_CONFIG_SCHEMA = (
    'empirical-lawhood/control/bounded-region-finite-anchor-observer-config'
)
MATCHED_ACTION_HOLD_OUTCOME_PROJECTION_RECEIPT_SCHEMA = (
    'empirical-lawhood/runtime/matched-action-hold-outcome-projection-evidence-receipt'
)
MATCHED_ACTION_HOLD_SEALED_EPISODE_SCHEMA = 'empirical-lawhood/runtime/matched-action-hold-sealed-episode'


class MatchedActionHoldUnitRole(StrEnum):
    EFFICACY_ACTIVE = "EFFICACY_ACTIVE"
    HOLD_CONTROL = "HOLD_CONTROL"


class MatchedActionHoldExecutionBranch(StrEnum):
    COMMITTED_CONTROLLER_ACTION = "COMMITTED_CONTROLLER_ACTION"
    QUALIFIED_HOLD_COUNTERFACTUAL = "QUALIFIED_HOLD_COUNTERFACTUAL"
    COMMITTED_HOLD_CONTROL = "COMMITTED_HOLD_CONTROL"


@dataclass(frozen=True, slots=True)
class MatchedActionHoldOutcomeProjectionEvidenceContract(CanonicalRecord):
    """Prospective binding for one substrate-owned outcome projector."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/planning/matched-action-hold-outcome-projection-evidence-contract'
    )

    contract_id: str
    projection_schema: str
    projection_specification: ObjectIdentity
    receipt_schema: str
    projector_implementation_id: str
    projector_implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.contract_id, field_name="contract_id")
        validate_schema(self.projection_schema)
        validate_schema(self.projection_specification.object_schema)
        validate_schema(self.receipt_schema)
        validate_stable_id(
            self.projector_implementation_id,
            field_name="projector_implementation_id",
        )
        validate_sha256(
            self.projector_implementation_sha256,
            field_name="projector_implementation_sha256",
        )
        if self.receipt_schema != MATCHED_ACTION_HOLD_OUTCOME_PROJECTION_RECEIPT_SCHEMA:
            raise ValueError("matched outcome projector uses another generic receipt schema")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldUnitSpec(CanonicalRecord):
    """One physical preparation; branches and members never add replication."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/matched-action-hold-unit-spec'

    unit_id: str
    physical_independent_unit_id: str
    preparation_coordinate_id: str
    preparation_sha256: str
    role: MatchedActionHoldUnitRole
    stratum_id: str | None
    hold_anchor_slot_id: str | None

    def __post_init__(self) -> None:
        for name, value in (
            ("unit_id", self.unit_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
            ("preparation_coordinate_id", self.preparation_coordinate_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(self.preparation_sha256, field_name="preparation_sha256")
        if self.role is MatchedActionHoldUnitRole.EFFICACY_ACTIVE:
            if self.stratum_id is None or self.hold_anchor_slot_id is not None:
                raise ValueError("matched efficacy unit requires only one stratum")
            validate_stable_id(self.stratum_id, field_name="stratum_id")
        else:
            if self.stratum_id is not None or self.hold_anchor_slot_id is None:
                raise ValueError("matched HOLD control requires only one anchor slot")
            validate_stable_id(
                self.hold_anchor_slot_id,
                field_name="hold_anchor_slot_id",
            )


@dataclass(frozen=True, slots=True)
class MatchedActionHoldExecutionCellSpec(CanonicalRecord):
    """One fresh member-local scientific execution nested in a unit."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/matched-action-hold-execution-cell-spec'

    episode_id: str
    execution_id: str
    environment_seed_id: str
    unit_id: str
    model_member_id: str
    branch: MatchedActionHoldExecutionBranch
    expected_action_word_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("episode_id", self.episode_id),
            ("execution_id", self.execution_id),
            ("environment_seed_id", self.environment_seed_id),
            ("unit_id", self.unit_id),
            ("model_member_id", self.model_member_id),
            ("expected_action_word_id", self.expected_action_word_id),
        ):
            validate_stable_id(value, field_name=name)

    @property
    def product_key(
        self,
    ) -> tuple[str, str, MatchedActionHoldExecutionBranch]:
        return (self.unit_id, self.model_member_id, self.branch)


@dataclass(frozen=True, slots=True)
class MatchedActionHoldAcquisitionPlanSeal(CanonicalRecord):
    """Outcome-free seal of the exact parameterised acquisition topology."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/matched-action-hold-acquisition-plan-seal'

    seal_id: str
    frozen_template: ObjectIdentity
    source_roster: ObjectIdentity
    execution_cells: tuple[MatchedActionHoldExecutionCellSpec, ...]
    acquisition_domain_id: str
    causal_cutoff_id: str
    causal_cutoff: ClockCoordinate
    source_episode_schema: str
    sealed_episode_schema: str
    projection_schema: str
    projection_receipt_schema: str
    expected_issued_manifest_schema: str
    expected_extension_set_schema: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for field_name, value in (
            ("seal_id", self.seal_id),
            ("acquisition_domain_id", self.acquisition_domain_id),
            ("causal_cutoff_id", self.causal_cutoff_id),
        ):
            validate_stable_id(value, field_name=field_name)
        if self.frozen_template.object_schema != MATCHED_ACTION_HOLD_EVALUATION_TEMPLATE_SCHEMA:
            raise ValueError("matched acquisition seal binds another template schema")
        require_sorted_unique_ids(
            self.execution_cells,
            attribute="episode_id",
            field_name="execution_cells",
        )
        for schema in (
            self.source_episode_schema,
            self.sealed_episode_schema,
            self.projection_schema,
            self.projection_receipt_schema,
            self.expected_issued_manifest_schema,
            self.expected_extension_set_schema,
        ):
            validate_schema(schema)
        if not self.execution_cells:
            raise ValueError("matched acquisition seal requires execution cells")
        if len({value.execution_id for value in self.execution_cells}) != len(
            self.execution_cells
        ) or len({value.environment_seed_id for value in self.execution_cells}) != len(
            self.execution_cells
        ):
            raise ValueError("matched acquisition seal reuses an execution or seed identity")
        if self.projection_receipt_schema != (
            MATCHED_ACTION_HOLD_OUTCOME_PROJECTION_RECEIPT_SCHEMA
        ):
            raise ValueError("matched acquisition seal uses another projection receipt")
        if self.sealed_episode_schema != MATCHED_ACTION_HOLD_SEALED_EPISODE_SCHEMA:
            raise ValueError("matched acquisition seal uses another sealed-episode schema")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("matched acquisition plan seal must remain outcome-free")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("matched acquisition plan seal must remain prospective")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldEfficacyReserveSpec(CanonicalRecord):
    """One predeclared whole-unit reserve for one efficacy stratum."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/matched-action-hold-efficacy-reserve-spec'

    reserve_id: str
    stratum_id: str
    preparation_coordinate_id: str
    preparation_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("reserve_id", self.reserve_id),
            ("stratum_id", self.stratum_id),
            ("preparation_coordinate_id", self.preparation_coordinate_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(self.preparation_sha256, field_name="preparation_sha256")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldUnitTemplate(CanonicalRecord):
    "One unit identity declared before measurement, whose preparation fingerprint is not yet known."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/matched-action-hold-unit-template'

    unit_id: str
    physical_independent_unit_id: str
    preparation_slot_id: str
    expected_preparation_sha256: str | None
    role: MatchedActionHoldUnitRole
    stratum_id: str | None
    hold_anchor_slot_id: str | None

    def __post_init__(self) -> None:
        for name, value in (
            ("unit_id", self.unit_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
            ("preparation_slot_id", self.preparation_slot_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.role is MatchedActionHoldUnitRole.EFFICACY_ACTIVE:
            if (
                self.stratum_id is None
                or self.hold_anchor_slot_id is not None
                or self.expected_preparation_sha256 is None
            ):
                raise ValueError("matched efficacy template requires only one stratum")
            validate_stable_id(self.stratum_id, field_name="stratum_id")
            validate_sha256(
                self.expected_preparation_sha256,
                field_name="expected_preparation_sha256",
            )
        else:
            if (
                self.stratum_id is not None
                or self.hold_anchor_slot_id is None
                or self.expected_preparation_sha256 is not None
            ):
                raise ValueError("matched HOLD template requires only one anchor slot")
            validate_stable_id(
                self.hold_anchor_slot_id,
                field_name="hold_anchor_slot_id",
            )


@dataclass(frozen=True, slots=True)
class MatchedActionHoldEfficacyReserveTemplate(CanonicalRecord):
    "One reserve slot declared before measurement, without a future preparation fingerprint."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/matched-action-hold-efficacy-reserve-template'

    reserve_id: str
    stratum_id: str
    preparation_slot_id: str
    expected_preparation_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("reserve_id", self.reserve_id),
            ("stratum_id", self.stratum_id),
            ("preparation_slot_id", self.preparation_slot_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(
            self.expected_preparation_sha256,
            field_name="expected_preparation_sha256",
        )


@dataclass(frozen=True, slots=True)
class MatchedActionHoldPreparationMaterialization(CanonicalRecord):
    """Outcome-blind attachment of one frozen slot to its exact preparation bytes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/matched-action-hold-preparation-materialization'

    materialization_id: str
    preparation_slot_id: str
    preparation_sha256: str
    selected_anchor_slot_id: str | None
    selected_anchor_selection_id: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.materialization_id, field_name="materialization_id")
        validate_stable_id(self.preparation_slot_id, field_name="preparation_slot_id")
        validate_sha256(self.preparation_sha256, field_name="preparation_sha256")
        anchor_bound = self.selected_anchor_slot_id is not None
        if anchor_bound != (self.selected_anchor_selection_id is not None):
            raise ValueError("matched preparation materialization has partial anchor lineage")
        if self.selected_anchor_slot_id is not None:
            validate_stable_id(
                self.selected_anchor_slot_id,
                field_name="selected_anchor_slot_id",
            )
        if self.selected_anchor_selection_id is not None:
            validate_stable_id(
                self.selected_anchor_selection_id,
                field_name="selected_anchor_selection_id",
            )


@dataclass(frozen=True, slots=True)
class MatchedActionHoldSelectedPreparationRoster(CanonicalRecord):
    "Post-selection materialization of every predeclared controller use preparation slot."

    SCHEMA: ClassVar[str] = MATCHED_ACTION_HOLD_SELECTED_PREPARATION_ROSTER_SCHEMA

    roster_id: str
    frozen_template: ObjectIdentity
    source_roster: ObjectIdentity
    unit_materializations: tuple[MatchedActionHoldPreparationMaterialization, ...]
    efficacy_reserve_materializations: tuple[
        MatchedActionHoldPreparationMaterialization,
        ...,
    ]
    native_hold_selected_roster: ObjectIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        if self.frozen_template.object_schema != MATCHED_ACTION_HOLD_EVALUATION_TEMPLATE_SCHEMA:
            raise ValueError("matched selected roster binds another template schema")
        if (
            self.native_hold_selected_roster.object_schema
            != NativeHoldCalibrationSelectedRoster.SCHEMA
        ):
            raise ValueError("matched selected roster binds another native-HOLD roster")
        require_sorted_unique_ids(
            self.unit_materializations,
            attribute="preparation_slot_id",
            field_name="unit_materializations",
        )
        require_sorted_unique_ids(
            self.efficacy_reserve_materializations,
            attribute="preparation_slot_id",
            field_name="efficacy_reserve_materializations",
        )
        all_values = (
            *self.unit_materializations,
            *self.efficacy_reserve_materializations,
        )
        if len({value.preparation_slot_id for value in all_values}) != len(all_values):
            raise ValueError("matched selected roster reuses a preparation slot")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("matched preparation selection must remain outcome-blind")


def _same_clock_domain(left: ClockCoordinate, right: ClockCoordinate) -> bool:
    return (
        left.clock_id == right.clock_id
        and left.time_unit == right.time_unit
        and left.coordinate_frame == right.coordinate_frame
        and left.origin is right.origin
    )


@dataclass(frozen=True, slots=True)
class MatchedActionHoldControllerEvaluationTemplate(CanonicalRecord):
    "Matched design declared before measurement, with future admission/HOLD fingerprints left unmaterialized."

    SCHEMA: ClassVar[str] = MATCHED_ACTION_HOLD_EVALUATION_TEMPLATE_SCHEMA

    template_id: str
    expected_evaluation_plan_id: str
    expected_precommitment_id: str
    expected_selected_preparation_roster_id: str
    expected_selected_preparation_roster_schema: str
    expected_source_roster_id: str
    expected_source_roster_schema: str
    expected_native_hold_selected_roster_id: str
    expected_native_hold_selected_roster_schema: str
    expected_native_hold_calibration_receipt_id: str
    expected_native_hold_calibration_receipt_schema: str
    expected_measured_hold_fibre_id: str
    expected_measured_hold_fibre_schema: str
    expected_finite_chart_reference_design_id: str
    expected_finite_chart_reference_design_schema: str
    expected_classifier_id: str
    expected_classifier_schema: str
    expected_issued_manifest_schema: str
    expected_extension_set_schema: str
    expected_acquisition_plan_seal_id: str
    expected_acquisition_plan_seal_schema: str
    expected_source_episode_schema: str
    expected_eligible_admission_study_id: str
    expected_eligible_admission_study_schema: str
    template_binder_implementation_id: str
    template_binder_implementation_sha256: str
    evaluator_boundary: EvaluatorBoundarySpec
    evaluator_binding: ImplementationBinding
    outcome_projection_evidence_contract: MatchedActionHoldOutcomeProjectionEvidenceContract
    units: tuple[MatchedActionHoldUnitTemplate, ...]
    execution_cells: tuple[MatchedActionHoldExecutionCellSpec, ...]
    model_member_ids: tuple[str, ...]
    qualified_action_word: OccurrenceActionWord
    qualified_hold_word: OccurrenceActionWord
    retained_history: ObjectIdentity
    receiver: ObjectIdentity
    horizon: ObjectIdentity
    active_decision_cell_ids: tuple[str, ...]
    hold_decision_cell_id: str
    strata: tuple[EvaluationStratumSpec, ...]
    efficacy_reserves: tuple[MatchedActionHoldEfficacyReserveTemplate, ...]
    hold_reserve_anchor_slot_id: str
    reserve_policy_id: str
    logical_roster_rule_id: str
    prospective_identity_domain_id: str
    causal_cutoff_id: str
    causal_cutoff: ClockCoordinate
    response_coordinates: tuple[ClockCoordinate, ...]
    effect_quantity_id: str
    effect_native_unit: str
    favorable_direction: UtilityDirection
    minimum_evaluable_efficacy_units: int
    minimum_active_coverage: Decimal
    minimum_reference_correctness_coverage: Decimal
    alpha: Decimal
    degrees_of_freedom: int
    one_sided_critical_value: Decimal
    materiality: NamedDecimal
    intent_to_treat: bool
    reduction_order: tuple[str, ...]
    mixed_efficacy_rule_id: str
    all_positive_strengthening_rule_id: str
    acquisition_domain_id: str
    reference_domain_id: str
    evaluation_domain_id: str
    maximum_claim_ceiling: str
    terminal_precedence: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        _validate_matched_action_hold_template(self)


def _validate_matched_action_hold_template(
    value: MatchedActionHoldControllerEvaluationTemplate,
) -> None:
    if not isinstance(
        value.outcome_projection_evidence_contract,
        MatchedActionHoldOutcomeProjectionEvidenceContract,
    ):
        raise TypeError("matched template requires a typed outcome-projection contract")
    for field_name, field_value in (
        ("template_id", value.template_id),
        ("expected_evaluation_plan_id", value.expected_evaluation_plan_id),
        ("expected_precommitment_id", value.expected_precommitment_id),
        (
            "expected_selected_preparation_roster_id",
            value.expected_selected_preparation_roster_id,
        ),
        ("expected_source_roster_id", value.expected_source_roster_id),
        (
            "expected_native_hold_selected_roster_id",
            value.expected_native_hold_selected_roster_id,
        ),
        (
            "expected_native_hold_calibration_receipt_id",
            value.expected_native_hold_calibration_receipt_id,
        ),
        ("expected_measured_hold_fibre_id", value.expected_measured_hold_fibre_id),
        (
            "expected_finite_chart_reference_design_id",
            value.expected_finite_chart_reference_design_id,
        ),
        ("expected_classifier_id", value.expected_classifier_id),
        (
            "expected_acquisition_plan_seal_id",
            value.expected_acquisition_plan_seal_id,
        ),
        (
            'expected_eligible_admission_study_id',
            value.expected_eligible_admission_study_id,
        ),
        (
            "template_binder_implementation_id",
            value.template_binder_implementation_id,
        ),
        ("hold_decision_cell_id", value.hold_decision_cell_id),
        ("hold_reserve_anchor_slot_id", value.hold_reserve_anchor_slot_id),
        ("reserve_policy_id", value.reserve_policy_id),
        ("logical_roster_rule_id", value.logical_roster_rule_id),
        ("prospective_identity_domain_id", value.prospective_identity_domain_id),
        ("causal_cutoff_id", value.causal_cutoff_id),
        ("effect_quantity_id", value.effect_quantity_id),
        ("mixed_efficacy_rule_id", value.mixed_efficacy_rule_id),
        (
            "all_positive_strengthening_rule_id",
            value.all_positive_strengthening_rule_id,
        ),
        ("acquisition_domain_id", value.acquisition_domain_id),
        ("reference_domain_id", value.reference_domain_id),
        ("evaluation_domain_id", value.evaluation_domain_id),
    ):
        validate_stable_id(field_value, field_name=field_name)
    validate_sha256(
        value.template_binder_implementation_sha256,
        field_name="template_binder_implementation_sha256",
    )
    validate_nonempty(value.effect_native_unit, field_name="effect_native_unit")
    validate_nonempty(value.maximum_claim_ceiling, field_name="maximum_claim_ceiling")
    schema_values = (
        value.expected_selected_preparation_roster_schema,
        value.expected_source_roster_schema,
        value.expected_native_hold_selected_roster_schema,
        value.expected_native_hold_calibration_receipt_schema,
        value.expected_measured_hold_fibre_schema,
        value.expected_finite_chart_reference_design_schema,
        value.expected_classifier_schema,
        value.expected_issued_manifest_schema,
        value.expected_extension_set_schema,
        value.expected_acquisition_plan_seal_schema,
        value.expected_source_episode_schema,
        value.expected_eligible_admission_study_schema,
    )
    for schema in schema_values:
        validate_schema(schema)
    if (
        value.expected_selected_preparation_roster_schema
        != MatchedActionHoldSelectedPreparationRoster.SCHEMA
        or value.expected_native_hold_selected_roster_schema
        != NativeHoldCalibrationSelectedRoster.SCHEMA
        or value.expected_native_hold_calibration_receipt_schema
        != NativeHoldDecisionCellCalibrationReceipt.SCHEMA
        or value.expected_measured_hold_fibre_schema != AdmissionMeasuredHoldFibre.SCHEMA
        or value.expected_finite_chart_reference_design_schema
        != FiniteChartReferenceDesign.SCHEMA
        or value.expected_classifier_schema != MATCHED_ACTION_HOLD_OBSERVER_CONFIG_SCHEMA
        or value.expected_eligible_admission_study_schema != AdmissionControllerStudy.SCHEMA
        or value.expected_acquisition_plan_seal_schema
        != MatchedActionHoldAcquisitionPlanSeal.SCHEMA
    ):
        raise ValueError("matched template expects another attachment after admission schema")

    require_sorted_unique_ids(value.units, attribute="unit_id", field_name="units")
    require_sorted_unique_ids(
        value.execution_cells,
        attribute="episode_id",
        field_name="execution_cells",
    )
    require_sorted_unique_strings(
        value.model_member_ids,
        field_name="model_member_ids",
        allow_empty=False,
    )
    require_sorted_unique_strings(
        value.active_decision_cell_ids,
        field_name="active_decision_cell_ids",
        allow_empty=False,
    )
    require_sorted_unique_ids(value.strata, attribute="stratum_id", field_name="strata")
    require_sorted_unique_ids(
        value.efficacy_reserves,
        attribute="reserve_id",
        field_name="efficacy_reserves",
    )
    efficacy = tuple(
        item for item in value.units if item.role is MatchedActionHoldUnitRole.EFFICACY_ACTIVE
    )
    controls = tuple(
        item for item in value.units if item.role is MatchedActionHoldUnitRole.HOLD_CONTROL
    )
    if len(efficacy) < 2 or not controls:
        raise ValueError(
            "matched template requires at least two efficacy units and one HOLD control"
        )
    if len({item.physical_independent_unit_id for item in value.units}) != len(value.units) or len(
        {item.preparation_slot_id for item in value.units}
    ) != len(value.units):
        raise ValueError("matched template reuses a physical unit or preparation slot")
    if (
        value.hold_decision_cell_id in value.active_decision_cell_ids
        or len({item.hold_anchor_slot_id for item in controls}) != len(controls)
        or value.hold_reserve_anchor_slot_id in {item.hold_anchor_slot_id for item in controls}
    ):
        raise ValueError("matched template HOLD cells or anchor slots alias active work")
    if not value.strata:
        raise ValueError("matched template requires efficacy strata")
    efficacy_by_id = {item.unit_id: item for item in efficacy}
    covered: set[str] = set()
    for stratum in value.strata:
        if (
            not stratum.independent_unit_ids
            or not 1 <= stratum.minimum_attempted_units <= len(stratum.independent_unit_ids)
            or any(
                efficacy_by_id.get(unit_id) is None
                or efficacy_by_id[unit_id].stratum_id != stratum.stratum_id
                for unit_id in stratum.independent_unit_ids
            )
        ):
            raise ValueError("matched template has an invalid efficacy stratum")
        covered.update(stratum.independent_unit_ids)
    if covered != set(efficacy_by_id):
        raise ValueError("matched template strata do not partition efficacy units")
    if (
        len(value.efficacy_reserves) != len(value.strata)
        or {item.stratum_id for item in value.efficacy_reserves}
        != {item.stratum_id for item in value.strata}
        or len({item.preparation_slot_id for item in value.efficacy_reserves})
        != len(value.efficacy_reserves)
        or any(
            item.preparation_slot_id in {unit.preparation_slot_id for unit in value.units}
            for item in value.efficacy_reserves
        )
    ):
        raise ValueError("matched template changes its whole-unit reserve slots")

    unit_by_id = {item.unit_id: item for item in value.units}
    expected_products: set[tuple[str, str, MatchedActionHoldExecutionBranch]] = set()
    for unit in value.units:
        branches = (
            (
                MatchedActionHoldExecutionBranch.COMMITTED_CONTROLLER_ACTION,
                MatchedActionHoldExecutionBranch.QUALIFIED_HOLD_COUNTERFACTUAL,
            )
            if unit.role is MatchedActionHoldUnitRole.EFFICACY_ACTIVE
            else (MatchedActionHoldExecutionBranch.COMMITTED_HOLD_CONTROL,)
        )
        expected_products.update(
            (unit.unit_id, member_id, branch)
            for member_id in value.model_member_ids
            for branch in branches
        )
    if (
        {item.product_key for item in value.execution_cells} != expected_products
        or len(value.execution_cells) != len(expected_products)
        or len({item.execution_id for item in value.execution_cells}) != len(value.execution_cells)
        or len({item.environment_seed_id for item in value.execution_cells})
        != len(value.execution_cells)
    ):
        raise ValueError("matched template execution product is incomplete or reused")
    for cell in value.execution_cells:
        unit = unit_by_id[cell.unit_id]
        expected_word_id = (
            value.qualified_action_word.word_id
            if cell.branch is MatchedActionHoldExecutionBranch.COMMITTED_CONTROLLER_ACTION
            else value.qualified_hold_word.word_id
        )
        if (
            cell.model_member_id not in value.model_member_ids
            or cell.expected_action_word_id != expected_word_id
            or (
                unit.role is MatchedActionHoldUnitRole.EFFICACY_ACTIVE
                and cell.branch is MatchedActionHoldExecutionBranch.COMMITTED_HOLD_CONTROL
            )
            or (
                unit.role is MatchedActionHoldUnitRole.HOLD_CONTROL
                and cell.branch is not MatchedActionHoldExecutionBranch.COMMITTED_HOLD_CONTROL
            )
        ):
            raise ValueError("matched template execution changes role/member/action")

    action = value.qualified_action_word
    hold = value.qualified_hold_word
    if (
        action.word_id == hold.word_id
        or action.support_status is not ActionWordSupportStatus.SUPPORTED
        or hold.support_status is not ActionWordSupportStatus.SUPPORTED
        or not action.occurrences
        or not hold.occurrences
        or (
            action.denominator_id,
            action.retained_history_id,
            action.receiver_id,
            action.horizon_id,
        )
        != (
            hold.denominator_id,
            hold.retained_history_id,
            hold.receiver_id,
            hold.horizon_id,
        )
        or value.retained_history.object_id != action.retained_history_id
        or value.receiver.object_id != action.receiver_id
        or value.horizon.object_id != action.horizon_id
    ):
        raise ValueError("matched template changes its exact action/HOLD semantics")
    if not value.response_coordinates or any(
        not _same_clock_domain(value.causal_cutoff, item)
        or item.coordinate <= value.causal_cutoff.coordinate
        for item in value.response_coordinates
    ):
        raise ValueError("matched template response coordinates change the causal horizon")
    coordinates = tuple(item.coordinate for item in value.response_coordinates)
    if coordinates != tuple(sorted(set(coordinates))):
        raise ValueError("matched template response coordinates must be sorted and unique")
    for field_name, decimal_value in (
        ("minimum_active_coverage", value.minimum_active_coverage),
        (
            "minimum_reference_correctness_coverage",
            value.minimum_reference_correctness_coverage,
        ),
        ("alpha", value.alpha),
        ("one_sided_critical_value", value.one_sided_critical_value),
    ):
        validate_decimal(decimal_value, field_name=field_name, minimum=Decimal(0))
    if (
        value.minimum_evaluable_efficacy_units != len(efficacy)
        or not Decimal(0) < value.minimum_active_coverage <= Decimal(1)
        or not Decimal(0) < value.minimum_reference_correctness_coverage <= Decimal(1)
        or not Decimal(0) < value.alpha < Decimal(1)
        or value.degrees_of_freedom != len(efficacy) - 1
        or value.one_sided_critical_value <= Decimal(0)
        or value.materiality.unit != value.effect_native_unit
        or value.materiality.value < Decimal(0)
        or value.favorable_direction is not UtilityDirection.HIGHER_IS_BETTER
        or not value.intent_to_treat
        or value.reduction_order != MATCHED_ACTION_HOLD_REDUCTION_ORDER
        or value.terminal_precedence != MATCHED_ACTION_HOLD_TERMINAL_PRECEDENCE
    ):
        raise ValueError("matched template has an incoherent frozen inference/terminal design")
    if value.effect_quantity_id not in value.evaluator_boundary.raw_outcome_quantity_ids:
        raise ValueError("matched template effect is absent from the evaluator boundary")
    predicate_quantity_ids = {
        item.quantity_id for item in value.evaluator_boundary.outcome_predicates
    }
    constraint_ids = tuple(
        constraint_id
        for item in value.evaluator_boundary.outcome_predicates
        for constraint_id in item.constraint_ids
    )
    if (
        set(value.evaluator_boundary.raw_outcome_quantity_ids)
        != predicate_quantity_ids | {value.effect_quantity_id}
        or any(not item.constraint_ids for item in value.evaluator_boundary.outcome_predicates)
        or len(set(constraint_ids)) != len(constraint_ids)
    ):
        raise ValueError("matched template changes the exact scalar/response constraint roster")
    if (
        value.evaluator_boundary.sealed_outcome_schema != MATCHED_ACTION_HOLD_SEALED_BUNDLE_SCHEMA
        or value.evaluator_boundary.revealed_outcome_schema
        != MATCHED_ACTION_HOLD_REVEALED_BUNDLE_SCHEMA
        or value.evaluator_binding.role is not ImplementationRole.OUTCOME_EVALUATOR
        or value.evaluator_boundary.outcome_evaluator_binding_id
        != value.evaluator_binding.binding_id
        or value.evaluator_binding.reference.input_schema
        != MATCHED_ACTION_HOLD_REVEALED_BUNDLE_SCHEMA
        or value.evaluator_binding.reference.output_schema
        != MATCHED_ACTION_HOLD_ADJUDICATION_SCHEMA
    ):
        raise ValueError("matched template changes the exact evaluator port")
    if (
        len(
            {
                value.acquisition_domain_id,
                value.reference_domain_id,
                value.evaluation_domain_id,
            }
        )
        != 3
    ):
        raise ValueError("matched template causal domains must remain disjoint")
    if (
        value.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        or value.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        or value.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
    ):
        raise ValueError("matched evaluation template must remain before measurement and outcome-blind")


def build_matched_action_hold_acquisition_plan_seal(
    *,
    template: MatchedActionHoldControllerEvaluationTemplate,
    source_roster: ObjectIdentity,
) -> MatchedActionHoldAcquisitionPlanSeal:
    """Seal the outcome-free parameterised acquisition topology before issuance.

    Only the required issued-manifest schema is prospective.  The actual
    hash-derived manifest identity is deliberately late-bound after issue so
    the seal can be included without a circular or future-outcome identity.
    """

    _require_expected_identity(
        source_roster,
        expected_id=template.expected_source_roster_id,
        expected_schema=template.expected_source_roster_schema,
        field_name="source roster",
    )
    contract = template.outcome_projection_evidence_contract
    return MatchedActionHoldAcquisitionPlanSeal(
        seal_id=template.expected_acquisition_plan_seal_id,
        frozen_template=ObjectIdentity.from_record(template.template_id, template),
        source_roster=source_roster,
        execution_cells=template.execution_cells,
        acquisition_domain_id=template.acquisition_domain_id,
        causal_cutoff_id=template.causal_cutoff_id,
        causal_cutoff=template.causal_cutoff,
        source_episode_schema=template.expected_source_episode_schema,
        sealed_episode_schema=MATCHED_ACTION_HOLD_SEALED_EPISODE_SCHEMA,
        projection_schema=contract.projection_schema,
        projection_receipt_schema=contract.receipt_schema,
        expected_issued_manifest_schema=template.expected_issued_manifest_schema,
        expected_extension_set_schema=template.expected_extension_set_schema,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


@dataclass(frozen=True, slots=True)
class MatchedActionHoldProspectiveAuthoringBundle(CanonicalRecord):
    """Issued-config root containing the template and its outcome-free seal."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/matched-action-hold-prospective-authoring-bundle'

    bundle_id: str
    template: MatchedActionHoldControllerEvaluationTemplate
    acquisition_plan_seal: MatchedActionHoldAcquisitionPlanSeal
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        if not isinstance(
            self.template,
            MatchedActionHoldControllerEvaluationTemplate,
        ):
            raise TypeError("matched controller use authoring bundle requires a typed template")
        if not isinstance(
            self.acquisition_plan_seal,
            MatchedActionHoldAcquisitionPlanSeal,
        ):
            raise TypeError("matched controller use authoring bundle requires a typed acquisition seal")
        expected_seal = build_matched_action_hold_acquisition_plan_seal(
            template=self.template,
            source_roster=self.acquisition_plan_seal.source_roster,
        )
        if self.acquisition_plan_seal != expected_seal:
            raise ValueError(
                "matched controller use authoring bundle acquisition seal differs from its template"
            )
        if (
            self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or self.template.outcome_access is not self.outcome_access
            or self.template.visibility_ceiling is not self.visibility_ceiling
        ):
            raise ValueError(
                "matched controller use authoring bundle must remain outcome-blind and prospective"
            )


def build_matched_action_hold_prospective_authoring_bundle(
    *,
    bundle_id: str,
    template: MatchedActionHoldControllerEvaluationTemplate,
    source_roster: ObjectIdentity,
) -> MatchedActionHoldProspectiveAuthoringBundle:
    """Freeze the template and exact acquisition topology as one config root."""

    return MatchedActionHoldProspectiveAuthoringBundle(
        bundle_id=bundle_id,
        template=template,
        acquisition_plan_seal=build_matched_action_hold_acquisition_plan_seal(
            template=template,
            source_roster=source_roster,
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


@dataclass(frozen=True, slots=True)
class MatchedActionHoldControllerEvaluationPlan(CanonicalRecord):
    "Exact parameterised matched-efficacy plus qualified-HOLD controller-use design."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/matched-action-hold-controller-evaluation-plan'

    evaluation_plan_id: str
    evaluator_boundary: EvaluatorBoundarySpec
    evaluator_binding: ImplementationBinding
    outcome_projection_evidence_contract: MatchedActionHoldOutcomeProjectionEvidenceContract
    units: tuple[MatchedActionHoldUnitSpec, ...]
    execution_cells: tuple[MatchedActionHoldExecutionCellSpec, ...]
    model_member_ids: tuple[str, ...]
    qualified_action_word: OccurrenceActionWord
    qualified_hold_word: OccurrenceActionWord
    measured_hold_fibre: ObjectIdentity
    native_hold_calibration: ObjectIdentity
    retained_history: ObjectIdentity
    receiver: ObjectIdentity
    horizon: ObjectIdentity
    finite_chart_reference_design: ObjectIdentity
    active_decision_cell_ids: tuple[str, ...]
    hold_decision_cell_id: str
    classifier: ObjectIdentity
    strata: tuple[EvaluationStratumSpec, ...]
    efficacy_reserves: tuple[MatchedActionHoldEfficacyReserveSpec, ...]
    hold_reserve_anchor_slot_id: str
    reserve_policy_id: str
    prospective_identity_domain_id: str
    causal_cutoff_id: str
    causal_cutoff: ClockCoordinate
    response_coordinates: tuple[ClockCoordinate, ...]
    effect_quantity_id: str
    effect_native_unit: str
    favorable_direction: UtilityDirection
    minimum_evaluable_efficacy_units: int
    minimum_active_coverage: Decimal
    minimum_reference_correctness_coverage: Decimal
    alpha: Decimal
    degrees_of_freedom: int
    one_sided_critical_value: Decimal
    materiality: NamedDecimal
    intent_to_treat: bool
    reduction_order: tuple[str, ...]
    mixed_efficacy_rule_id: str
    all_positive_strengthening_rule_id: str
    acquisition_domain_id: str
    reference_domain_id: str
    evaluation_domain_id: str
    maximum_claim_ceiling: str
    terminal_precedence: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        if not isinstance(
            self.outcome_projection_evidence_contract,
            MatchedActionHoldOutcomeProjectionEvidenceContract,
        ):
            raise TypeError("matched plan requires a typed outcome-projection contract")
        for name, value in (
            ("evaluation_plan_id", self.evaluation_plan_id),
            ("hold_decision_cell_id", self.hold_decision_cell_id),
            ("hold_reserve_anchor_slot_id", self.hold_reserve_anchor_slot_id),
            ("reserve_policy_id", self.reserve_policy_id),
            ("prospective_identity_domain_id", self.prospective_identity_domain_id),
            ("causal_cutoff_id", self.causal_cutoff_id),
            ("effect_quantity_id", self.effect_quantity_id),
            ("mixed_efficacy_rule_id", self.mixed_efficacy_rule_id),
            (
                "all_positive_strengthening_rule_id",
                self.all_positive_strengthening_rule_id,
            ),
            ("acquisition_domain_id", self.acquisition_domain_id),
            ("reference_domain_id", self.reference_domain_id),
            ("evaluation_domain_id", self.evaluation_domain_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.effect_native_unit, field_name="effect_native_unit")
        validate_nonempty(
            self.maximum_claim_ceiling,
            field_name="maximum_claim_ceiling",
        )
        require_sorted_unique_ids(
            self.units,
            attribute="unit_id",
            field_name="units",
        )
        require_sorted_unique_ids(
            self.execution_cells,
            attribute="episode_id",
            field_name="execution_cells",
        )
        require_sorted_unique_strings(
            self.model_member_ids,
            field_name="model_member_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.active_decision_cell_ids,
            field_name="active_decision_cell_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.strata,
            attribute="stratum_id",
            field_name="strata",
        )
        require_sorted_unique_ids(
            self.efficacy_reserves,
            attribute="reserve_id",
            field_name="efficacy_reserves",
        )
        for decision_cell_id in self.active_decision_cell_ids:
            validate_stable_id(decision_cell_id, field_name="active_decision_cell_ids")
        if self.hold_decision_cell_id in self.active_decision_cell_ids:
            raise ValueError("matched HOLD-only cell aliases active support")
        efficacy_units = tuple(
            value
            for value in self.units
            if value.role is MatchedActionHoldUnitRole.EFFICACY_ACTIVE
        )
        control_units = tuple(
            value for value in self.units if value.role is MatchedActionHoldUnitRole.HOLD_CONTROL
        )
        if len(efficacy_units) < 2 or not control_units:
            raise ValueError("matched controller use requires at least two efficacy units and one control")
        if len({value.physical_independent_unit_id for value in self.units}) != len(self.units):
            raise ValueError("matched controller use reuses a physical independent-unit identity")
        if len({value.preparation_coordinate_id for value in self.units}) != len(self.units):
            raise ValueError("matched controller use scientific units reuse a preparation coordinate")
        if len({value.hold_anchor_slot_id for value in control_units}) != len(control_units):
            raise ValueError("matched HOLD controls reuse an anchor slot")

        if not self.strata:
            raise ValueError("matched controller use requires efficacy strata")
        efficacy_ids = {value.unit_id for value in efficacy_units}
        unit_strata = {value.unit_id: value.stratum_id for value in efficacy_units}
        covered: set[str] = set()
        for stratum in self.strata:
            if not stratum.independent_unit_ids or not (
                1 <= stratum.minimum_attempted_units <= len(stratum.independent_unit_ids)
            ):
                raise ValueError("matched controller use stratum has an invalid attempted-unit threshold")
            if any(
                unit_strata.get(unit_id) != stratum.stratum_id
                for unit_id in stratum.independent_unit_ids
            ):
                raise ValueError("matched controller use stratum assignment changes a unit")
            covered.update(stratum.independent_unit_ids)
        if covered != efficacy_ids:
            raise ValueError("matched controller use strata do not partition the efficacy roster")

        if len(self.efficacy_reserves) != len(self.strata) or {
            value.stratum_id for value in self.efficacy_reserves
        } != {value.stratum_id for value in self.strata}:
            raise ValueError("matched controller use requires one efficacy reserve per stratum")
        if len({value.preparation_coordinate_id for value in self.efficacy_reserves}) != len(
            self.efficacy_reserves
        ) or len({value.preparation_sha256 for value in self.efficacy_reserves}) != len(
            self.efficacy_reserves
        ):
            raise ValueError("matched efficacy reserves reuse preparation identity")
        occupied_coordinates = {value.preparation_coordinate_id for value in self.units}
        occupied_hashes = {value.preparation_sha256 for value in self.units}
        if any(
            value.preparation_coordinate_id in occupied_coordinates
            or value.preparation_sha256 in occupied_hashes
            for value in self.efficacy_reserves
        ):
            raise ValueError("matched efficacy reserve aliases a scientific unit")
        control_anchor_ids = {value.hold_anchor_slot_id for value in control_units}
        if self.hold_reserve_anchor_slot_id in control_anchor_ids:
            raise ValueError("matched HOLD reserve aliases a scheduled control")

        units = {value.unit_id: value for value in self.units}
        expected_products: set[tuple[str, str, MatchedActionHoldExecutionBranch]] = set()
        for unit in self.units:
            branches = (
                (
                    MatchedActionHoldExecutionBranch.COMMITTED_CONTROLLER_ACTION,
                    MatchedActionHoldExecutionBranch.QUALIFIED_HOLD_COUNTERFACTUAL,
                )
                if unit.role is MatchedActionHoldUnitRole.EFFICACY_ACTIVE
                else (MatchedActionHoldExecutionBranch.COMMITTED_HOLD_CONTROL,)
            )
            expected_products.update(
                (unit.unit_id, member_id, branch)
                for member_id in self.model_member_ids
                for branch in branches
            )
        observed_products = {value.product_key for value in self.execution_cells}
        if observed_products != expected_products or len(self.execution_cells) != len(
            expected_products
        ):
            raise ValueError("matched controller use execution product is incomplete or contains extras")
        if len({value.execution_id for value in self.execution_cells}) != len(
            self.execution_cells
        ) or len({value.environment_seed_id for value in self.execution_cells}) != len(
            self.execution_cells
        ):
            raise ValueError("matched controller use executions reuse an execution or seed identity")
        for cell in self.execution_cells:
            unit = units[cell.unit_id]
            if cell.model_member_id not in self.model_member_ids:
                raise ValueError("matched execution uses an undeclared member")
            if (
                unit.role is MatchedActionHoldUnitRole.EFFICACY_ACTIVE
                and cell.branch is MatchedActionHoldExecutionBranch.COMMITTED_HOLD_CONTROL
            ) or (
                unit.role is MatchedActionHoldUnitRole.HOLD_CONTROL
                and cell.branch is not MatchedActionHoldExecutionBranch.COMMITTED_HOLD_CONTROL
            ):
                raise ValueError("matched execution branch changes its unit role")
            expected_word_id = (
                self.qualified_action_word.word_id
                if cell.branch is MatchedActionHoldExecutionBranch.COMMITTED_CONTROLLER_ACTION
                else self.qualified_hold_word.word_id
            )
            if cell.expected_action_word_id != expected_word_id:
                raise ValueError("matched execution branch uses another action word")

        action = self.qualified_action_word
        hold = self.qualified_hold_word
        if (
            action.word_id == hold.word_id
            or action.support_status is not ActionWordSupportStatus.SUPPORTED
            or hold.support_status is not ActionWordSupportStatus.SUPPORTED
            or not action.occurrences
            or not hold.occurrences
        ):
            raise ValueError("matched controller use requires distinct supported action and HOLD words")
        if (
            action.denominator_id,
            action.retained_history_id,
            action.receiver_id,
            action.horizon_id,
        ) != (
            hold.denominator_id,
            hold.retained_history_id,
            hold.receiver_id,
            hold.horizon_id,
        ):
            raise ValueError("matched action and HOLD change D/H/R/horizon")
        if (
            self.retained_history.object_id != action.retained_history_id
            or self.receiver.object_id != action.receiver_id
            or self.horizon.object_id != action.horizon_id
        ):
            raise ValueError("matched plan identities differ from action semantics")
        if self.measured_hold_fibre.object_schema != ('empirical-lawhood/planning/admission-measured-hold-fibre'):
            raise ValueError("matched controller use requires a measured-HOLD fibre")
        if self.native_hold_calibration.object_schema != (
            NativeHoldDecisionCellCalibrationReceipt.SCHEMA
        ):
            raise ValueError("matched controller use requires the finite native-HOLD calibration")
        if self.finite_chart_reference_design.object_schema != (
            'empirical-lawhood/planning/finite-chart-reference-design'
        ):
            raise ValueError("matched controller use requires the finite-chart reference design")

        if not self.response_coordinates:
            raise ValueError("matched controller use requires receiver response coordinates")
        if any(
            not _same_clock_domain(self.causal_cutoff, value)
            or value.coordinate <= self.causal_cutoff.coordinate
            for value in self.response_coordinates
        ):
            raise ValueError("matched receiver coordinates do not follow the causal cutoff")
        coordinates = tuple(value.coordinate for value in self.response_coordinates)
        if coordinates != tuple(sorted(set(coordinates))):
            raise ValueError("matched receiver coordinates must be sorted and unique")

        for decimal_name, decimal_value in (
            ("minimum_active_coverage", self.minimum_active_coverage),
            (
                "minimum_reference_correctness_coverage",
                self.minimum_reference_correctness_coverage,
            ),
            ("alpha", self.alpha),
            ("one_sided_critical_value", self.one_sided_critical_value),
        ):
            validate_decimal(
                decimal_value,
                field_name=decimal_name,
                minimum=Decimal(0),
            )
        if (
            self.minimum_evaluable_efficacy_units != len(efficacy_units)
            or not Decimal(0) < self.minimum_active_coverage <= Decimal(1)
            or not Decimal(0) < self.minimum_reference_correctness_coverage <= Decimal(1)
            or not Decimal(0) < self.alpha < Decimal(1)
            or self.degrees_of_freedom != len(efficacy_units) - 1
            or self.one_sided_critical_value <= Decimal(0)
        ):
            raise ValueError("matched controller use inference constants are incoherent with its roster")
        if self.materiality.unit != self.effect_native_unit or self.materiality.value < Decimal(0):
            raise ValueError("matched controller use materiality is invalid for the native effect")
        if self.favorable_direction is not UtilityDirection.HIGHER_IS_BETTER:
            raise ValueError("matched controller use requires action-minus-HOLD efficacy")
        if self.effect_quantity_id not in self.evaluator_boundary.raw_outcome_quantity_ids:
            raise ValueError("matched effect quantity is absent from the reveal boundary")
        predicate_quantity_ids = {
            item.quantity_id for item in self.evaluator_boundary.outcome_predicates
        }
        constraint_ids = tuple(
            constraint_id
            for item in self.evaluator_boundary.outcome_predicates
            for constraint_id in item.constraint_ids
        )
        if (
            set(self.evaluator_boundary.raw_outcome_quantity_ids)
            != predicate_quantity_ids | {self.effect_quantity_id}
            or any(not item.constraint_ids for item in self.evaluator_boundary.outcome_predicates)
            or len(set(constraint_ids)) != len(constraint_ids)
        ):
            raise ValueError("matched plan changes the exact scalar/response constraint roster")
        if (
            self.evaluator_boundary.sealed_outcome_schema
            != MATCHED_ACTION_HOLD_SEALED_BUNDLE_SCHEMA
            or self.evaluator_boundary.revealed_outcome_schema
            != MATCHED_ACTION_HOLD_REVEALED_BUNDLE_SCHEMA
        ):
            raise ValueError("matched evaluator boundary uses another bundle contract")
        if (
            self.evaluator_binding.role is not ImplementationRole.OUTCOME_EVALUATOR
            or self.evaluator_boundary.outcome_evaluator_binding_id
            != self.evaluator_binding.binding_id
            or self.evaluator_binding.reference.input_schema
            != MATCHED_ACTION_HOLD_REVEALED_BUNDLE_SCHEMA
            or self.evaluator_binding.reference.output_schema
            != MATCHED_ACTION_HOLD_ADJUDICATION_SCHEMA
        ):
            raise ValueError("matched controller use evaluator binding uses another exact port")
        if self.reduction_order != MATCHED_ACTION_HOLD_REDUCTION_ORDER:
            raise ValueError("matched controller use reducer order differs from the frozen order")
        if self.terminal_precedence != MATCHED_ACTION_HOLD_TERMINAL_PRECEDENCE:
            raise ValueError("matched controller use terminal precedence differs from the frozen matrix")
        if not self.intent_to_treat:
            raise ValueError("matched controller use requires complete ITT accounting")
        if (
            len(
                {
                    self.acquisition_domain_id,
                    self.reference_domain_id,
                    self.evaluation_domain_id,
                }
            )
            != 3
        ):
            raise ValueError("matched acquisition/reference/evaluation domains must be disjoint")
        if self.evidence_ceiling is not EvidenceCeiling.CONTROLLER_USE:
            raise ValueError("matched controller evaluation must remain controller use")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("matched controller use plan must remain sealed")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("matched controller use plan must remain prospective")

    @property
    def efficacy_units(self) -> tuple[MatchedActionHoldUnitSpec, ...]:
        return tuple(
            value
            for value in self.units
            if value.role is MatchedActionHoldUnitRole.EFFICACY_ACTIVE
        )

    @property
    def hold_controls(self) -> tuple[MatchedActionHoldUnitSpec, ...]:
        return tuple(
            value for value in self.units if value.role is MatchedActionHoldUnitRole.HOLD_CONTROL
        )


@dataclass(frozen=True, slots=True)
class MatchedActionHoldEvaluationPrecommitment(CanonicalRecord):
    "Materialization after admission of one issued, prospectively sealed controller-use bundle."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/matched-action-hold-evaluation-precommitment'

    precommitment_id: str
    authoring_bundle: ObjectIdentity
    frozen_template: ObjectIdentity
    selected_preparation_roster: ObjectIdentity
    native_hold_selected_roster: ObjectIdentity
    eligible_admission_study: ObjectIdentity
    issued_manifest: ObjectIdentity
    extension_set: ObjectIdentity
    acquisition_plan_seal: ObjectIdentity
    matched_plan: MatchedActionHoldControllerEvaluationPlan
    reference_design: ObjectIdentity
    evaluator_binding: ImplementationBinding
    logical_roster_rule_id: str
    causal_cutoff_id: str
    acquisition_domain_id: str
    reference_domain_id: str
    evaluation_domain_id: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.precommitment_id, field_name="precommitment_id")
        expected_schemas = (
            (
                self.authoring_bundle,
                MatchedActionHoldProspectiveAuthoringBundle.SCHEMA,
            ),
            (self.frozen_template, MatchedActionHoldControllerEvaluationTemplate.SCHEMA),
            (
                self.selected_preparation_roster,
                MatchedActionHoldSelectedPreparationRoster.SCHEMA,
            ),
            (
                self.native_hold_selected_roster,
                NativeHoldCalibrationSelectedRoster.SCHEMA,
            ),
            (self.eligible_admission_study, AdmissionControllerStudy.SCHEMA),
            (
                self.acquisition_plan_seal,
                MatchedActionHoldAcquisitionPlanSeal.SCHEMA,
            ),
        )
        if any(identity.object_schema != schema for identity, schema in expected_schemas):
            raise ValueError("matched precommitment contains another materialization schema")
        for name, value in (
            ("logical_roster_rule_id", self.logical_roster_rule_id),
            ("causal_cutoff_id", self.causal_cutoff_id),
            ("acquisition_domain_id", self.acquisition_domain_id),
            ("reference_domain_id", self.reference_domain_id),
            ("evaluation_domain_id", self.evaluation_domain_id),
        ):
            validate_stable_id(value, field_name=name)
        if (
            len(
                {
                    self.acquisition_domain_id,
                    self.reference_domain_id,
                    self.evaluation_domain_id,
                }
            )
            != 3
        ):
            raise ValueError("matched prospective causal domains must be disjoint")
        plan = self.matched_plan
        if (
            self.reference_design != plan.finite_chart_reference_design
            or self.evaluator_binding != plan.evaluator_binding
            or self.reference_domain_id != plan.reference_domain_id
            or self.evaluation_domain_id != plan.evaluation_domain_id
            or self.causal_cutoff_id != plan.causal_cutoff_id
            or self.acquisition_domain_id != plan.acquisition_domain_id
        ):
            raise ValueError("matched precommitment changes its frozen plan")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("matched precommitment must remain sealed")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("matched precommitment cannot be outcome-visible")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldEvaluationTemplateBindingReceipt(CanonicalRecord):
    "Deterministic attachment after admission of a matched template declared before measurement."

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/planning/matched-action-hold-evaluation-template-binding-receipt'
    )

    receipt_id: str
    authoring_bundle: ObjectIdentity
    template: ObjectIdentity
    selected_preparation_roster: ObjectIdentity
    native_hold_selected_roster: ObjectIdentity
    native_hold_calibration: ObjectIdentity
    measured_hold_fibre: ObjectIdentity
    finite_chart_reference_design: ObjectIdentity
    classifier: ObjectIdentity
    eligible_admission_study: ObjectIdentity
    issued_manifest: ObjectIdentity
    extension_set: ObjectIdentity
    acquisition_plan_seal: ObjectIdentity
    evaluation_plan: ObjectIdentity
    precommitment: ObjectIdentity
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
        validate_sha256(
            self.binding_implementation_sha256,
            field_name="binding_implementation_sha256",
        )
        expected_schemas = (
            (
                self.authoring_bundle,
                MatchedActionHoldProspectiveAuthoringBundle.SCHEMA,
            ),
            (self.template, MatchedActionHoldControllerEvaluationTemplate.SCHEMA),
            (
                self.selected_preparation_roster,
                MatchedActionHoldSelectedPreparationRoster.SCHEMA,
            ),
            (
                self.native_hold_selected_roster,
                NativeHoldCalibrationSelectedRoster.SCHEMA,
            ),
            (
                self.native_hold_calibration,
                NativeHoldDecisionCellCalibrationReceipt.SCHEMA,
            ),
            (self.measured_hold_fibre, AdmissionMeasuredHoldFibre.SCHEMA),
            (
                self.finite_chart_reference_design,
                FiniteChartReferenceDesign.SCHEMA,
            ),
            (self.classifier, MATCHED_ACTION_HOLD_OBSERVER_CONFIG_SCHEMA),
            (self.eligible_admission_study, AdmissionControllerStudy.SCHEMA),
            (
                self.acquisition_plan_seal,
                MatchedActionHoldAcquisitionPlanSeal.SCHEMA,
            ),
            (self.evaluation_plan, MatchedActionHoldControllerEvaluationPlan.SCHEMA),
            (self.precommitment, MatchedActionHoldEvaluationPrecommitment.SCHEMA),
        )
        if any(identity.object_schema != schema for identity, schema in expected_schemas):
            raise ValueError("matched template binding contains another record schema")
        if self.outcome_dependent_choice or self.scientific_verdict is not None:
            raise ValueError("matched template binding cannot select or adjudicate outcomes")
        if (
            self.evidence_ceiling is not EvidenceCeiling.ADMISSION
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError(
                "matched template binding must remain an outcome-blind attachment after admission"
            )


def _require_expected_identity(
    identity: ObjectIdentity,
    *,
    expected_id: str,
    expected_schema: str,
    field_name: str,
) -> None:
    if identity.object_id != expected_id or identity.object_schema != expected_schema:
        raise ValueError(f"matched template binding substitutes {field_name}")


def bind_matched_action_hold_controller_evaluation_template(
    *,
    receipt_id: str,
    authoring_bundle: MatchedActionHoldProspectiveAuthoringBundle,
    selected_preparation_roster: MatchedActionHoldSelectedPreparationRoster,
    native_hold_selected_roster: NativeHoldCalibrationSelectedRoster,
    native_hold_calibration: NativeHoldDecisionCellCalibrationReceipt,
    measured_hold_fibre: AdmissionMeasuredHoldFibre,
    finite_chart_reference_design: FiniteChartReferenceDesign,
    classifier: ObjectIdentity,
    eligible_admission_study: AdmissionControllerStudy,
    issued_manifest: ObjectIdentity,
    extension_set: ObjectIdentity,
    binding_implementation_id: str,
    binding_implementation_sha256: str,
) -> tuple[
    MatchedActionHoldControllerEvaluationPlan,
    MatchedActionHoldEvaluationPrecommitment,
    MatchedActionHoldEvaluationTemplateBindingReceipt,
]:
    "Attach exact eligible admission records without making a design choice after measurement."

    if not isinstance(
        authoring_bundle,
        MatchedActionHoldProspectiveAuthoringBundle,
    ):
        raise TypeError("matched template binding requires a typed authoring bundle")
    template = authoring_bundle.template
    acquisition_plan_seal = authoring_bundle.acquisition_plan_seal
    if (
        binding_implementation_id != template.template_binder_implementation_id
        or binding_implementation_sha256 != template.template_binder_implementation_sha256
    ):
        raise ValueError("matched template binding uses another frozen implementation")
    template_identity = ObjectIdentity.from_record(template.template_id, template)
    authoring_bundle_identity = ObjectIdentity.from_record(
        authoring_bundle.bundle_id,
        authoring_bundle,
    )
    selected_roster_identity = ObjectIdentity.from_record(
        selected_preparation_roster.roster_id,
        selected_preparation_roster,
    )
    native_roster_identity = ObjectIdentity.from_record(
        native_hold_selected_roster.roster_id,
        native_hold_selected_roster,
    )
    calibration_identity = ObjectIdentity.from_record(
        native_hold_calibration.receipt_id,
        native_hold_calibration,
    )
    measured_hold_identity = ObjectIdentity.from_record(
        measured_hold_fibre.hold_fibre_id,
        measured_hold_fibre,
    )
    reference_identity = ObjectIdentity.from_record(
        finite_chart_reference_design.reference_design_id,
        finite_chart_reference_design,
    )
    programme_identity = ObjectIdentity.from_record(
        eligible_admission_study.study_id,
        eligible_admission_study,
    )
    acquisition_plan_seal_identity = ObjectIdentity.from_record(
        acquisition_plan_seal.seal_id,
        acquisition_plan_seal,
    )
    for identity, expected_id, expected_schema, field_name in (
        (
            selected_roster_identity,
            template.expected_selected_preparation_roster_id,
            template.expected_selected_preparation_roster_schema,
            "selected preparation roster",
        ),
        (
            selected_preparation_roster.source_roster,
            template.expected_source_roster_id,
            template.expected_source_roster_schema,
            "source roster",
        ),
        (
            native_roster_identity,
            template.expected_native_hold_selected_roster_id,
            template.expected_native_hold_selected_roster_schema,
            "native-HOLD selected roster",
        ),
        (
            calibration_identity,
            template.expected_native_hold_calibration_receipt_id,
            template.expected_native_hold_calibration_receipt_schema,
            "native-HOLD calibration",
        ),
        (
            measured_hold_identity,
            template.expected_measured_hold_fibre_id,
            template.expected_measured_hold_fibre_schema,
            "measured-HOLD fibre",
        ),
        (
            reference_identity,
            template.expected_finite_chart_reference_design_id,
            template.expected_finite_chart_reference_design_schema,
            "finite-chart reference",
        ),
        (
            classifier,
            template.expected_classifier_id,
            template.expected_classifier_schema,
            "classifier",
        ),
        (
            programme_identity,
            template.expected_eligible_admission_study_id,
            template.expected_eligible_admission_study_schema,
            "eligible controller admission programme",
        ),
        (
            acquisition_plan_seal_identity,
            template.expected_acquisition_plan_seal_id,
            template.expected_acquisition_plan_seal_schema,
            "acquisition plan seal",
        ),
    ):
        _require_expected_identity(
            identity,
            expected_id=expected_id,
            expected_schema=expected_schema,
            field_name=field_name,
        )
    if issued_manifest.object_schema != template.expected_issued_manifest_schema:
        raise ValueError("matched template binding substitutes issued manifest schema")
    if extension_set.object_schema != template.expected_extension_set_schema:
        raise ValueError("matched template binding substitutes extension-set schema")
    if (
        selected_preparation_roster.frozen_template != template_identity
        or selected_preparation_roster.native_hold_selected_roster != native_roster_identity
    ):
        raise ValueError("matched preparation roster changes template/HOLD lineage")
    expected_acquisition_plan_seal = build_matched_action_hold_acquisition_plan_seal(
        template=template,
        source_roster=selected_preparation_roster.source_roster,
    )
    if acquisition_plan_seal != expected_acquisition_plan_seal:
        raise ValueError("matched acquisition plan seal changes the prospective topology")
    if (
        acquisition_plan_seal.expected_issued_manifest_schema != issued_manifest.object_schema
        or acquisition_plan_seal.expected_extension_set_schema != extension_set.object_schema
    ):
        raise ValueError("matched issued schemas differ from the acquisition plan seal")

    unit_materializations = {
        item.preparation_slot_id: item for item in selected_preparation_roster.unit_materializations
    }
    reserve_materializations = {
        item.preparation_slot_id: item
        for item in selected_preparation_roster.efficacy_reserve_materializations
    }
    if set(unit_materializations) != {item.preparation_slot_id for item in template.units} or set(
        reserve_materializations
    ) != {item.preparation_slot_id for item in template.efficacy_reserves}:
        raise ValueError("matched selected roster omits or adds a frozen preparation slot")
    selected_anchors = {item.slot_id: item for item in native_hold_selected_roster.selections}
    expected_anchor_slots = {
        item.hold_anchor_slot_id
        for item in template.units
        if item.role is MatchedActionHoldUnitRole.HOLD_CONTROL
    } | {template.hold_reserve_anchor_slot_id}
    if set(selected_anchors) != expected_anchor_slots:
        raise ValueError("matched native-HOLD roster changes control/reserve anchor slots")
    for unit in template.units:
        materialization = unit_materializations[unit.preparation_slot_id]
        if unit.role is MatchedActionHoldUnitRole.EFFICACY_ACTIVE:
            if (
                materialization.selected_anchor_slot_id is not None
                or materialization.selected_anchor_selection_id is not None
                or materialization.preparation_sha256 != unit.expected_preparation_sha256
            ):
                raise ValueError(
                    "matched efficacy preparation changes fingerprint or anchor lineage"
                )
            continue
        slot_id = unit.hold_anchor_slot_id
        if slot_id is None:  # pragma: no cover - unit-template invariant
            raise AssertionError("matched HOLD unit lost its anchor slot")
        selected = selected_anchors[slot_id]
        if (
            materialization.selected_anchor_slot_id != slot_id
            or materialization.selected_anchor_selection_id != selected.selection_id
            or materialization.preparation_sha256 != selected.selected.preparation_fingerprint
        ):
            raise ValueError("matched HOLD materialization differs from selected anchor")
    reserve_templates = {item.preparation_slot_id: item for item in template.efficacy_reserves}
    if any(
        item.selected_anchor_slot_id is not None
        or item.selected_anchor_selection_id is not None
        or item.preparation_sha256
        != reserve_templates[item.preparation_slot_id].expected_preparation_sha256
        for item in reserve_materializations.values()
    ):
        raise ValueError("matched efficacy reserve changes fingerprint or anchor lineage")

    if (
        native_hold_calibration.selected_roster != native_roster_identity
        or native_hold_calibration.disposition is not NativeHoldCalibrationDisposition.SUPPORTED
        or native_hold_calibration.hold_decision_cell_id != template.hold_decision_cell_id
        or native_hold_calibration.active_decision_cell_ids != template.active_decision_cell_ids
        or measured_hold_fibre.calibration_id != native_hold_calibration.receipt_id
        or measured_hold_fibre.decision_cell_id != template.hold_decision_cell_id
        or native_hold_calibration.selected_sibling_hold_candidate_key_id
        != measured_hold_fibre.admission_candidate_cell.object_id
    ):
        raise ValueError("matched HOLD fibre/calibration/decision-cell lineage differs")
    calibration_members = {
        item.model_member_id for item in native_hold_calibration.expected_entries
    }
    if calibration_members != set(template.model_member_ids):
        raise ValueError("matched HOLD calibration changes the numerical-member roster")

    programme_words = {
        item.action_word.word_id: item.action_word for item in eligible_admission_study.action_bindings
    }
    active_cells = tuple(
        sorted(
            {
                item.decision_cell_id
                for item in eligible_admission_study.synthesis.candidate_chart.candidates
            }
        )
    )
    if (
        eligible_admission_study.prospective_evaluation is not None
        or any(
            item.role is ImplementationRole.OUTCOME_EVALUATOR
            for item in eligible_admission_study.implementations
        )
        or eligible_admission_study.measured_hold_fibre != measured_hold_fibre
        or programme_words.get(template.qualified_action_word.word_id)
        != template.qualified_action_word
        or programme_words.get(template.qualified_hold_word.word_id) != template.qualified_hold_word
        or active_cells != template.active_decision_cell_ids
    ):
        raise ValueError("matched template binding uses another programme eligible for admission")

    reference_words = {item.word_id: item for item in finite_chart_reference_design.action_words}
    if (
        finite_chart_reference_design.model_member_ids != template.model_member_ids
        or reference_words.get(template.qualified_action_word.word_id)
        != template.qualified_action_word
        or reference_words.get(template.qualified_hold_word.word_id) != template.qualified_hold_word
        or finite_chart_reference_design.measured_hold_word_id
        != template.qualified_hold_word.word_id
    ):
        raise ValueError("matched finite-chart reference changes member/action/HOLD identity")

    units = tuple(
        MatchedActionHoldUnitSpec(
            unit_id=item.unit_id,
            physical_independent_unit_id=item.physical_independent_unit_id,
            preparation_coordinate_id=item.preparation_slot_id,
            preparation_sha256=unit_materializations[item.preparation_slot_id].preparation_sha256,
            role=item.role,
            stratum_id=item.stratum_id,
            hold_anchor_slot_id=item.hold_anchor_slot_id,
        )
        for item in template.units
    )
    reserves = tuple(
        MatchedActionHoldEfficacyReserveSpec(
            reserve_id=item.reserve_id,
            stratum_id=item.stratum_id,
            preparation_coordinate_id=item.preparation_slot_id,
            preparation_sha256=reserve_materializations[
                item.preparation_slot_id
            ].preparation_sha256,
        )
        for item in template.efficacy_reserves
    )
    plan = MatchedActionHoldControllerEvaluationPlan(
        evaluation_plan_id=template.expected_evaluation_plan_id,
        evaluator_boundary=template.evaluator_boundary,
        evaluator_binding=template.evaluator_binding,
        outcome_projection_evidence_contract=(template.outcome_projection_evidence_contract),
        units=units,
        execution_cells=template.execution_cells,
        model_member_ids=template.model_member_ids,
        qualified_action_word=template.qualified_action_word,
        qualified_hold_word=template.qualified_hold_word,
        measured_hold_fibre=measured_hold_identity,
        native_hold_calibration=calibration_identity,
        retained_history=template.retained_history,
        receiver=template.receiver,
        horizon=template.horizon,
        finite_chart_reference_design=reference_identity,
        active_decision_cell_ids=template.active_decision_cell_ids,
        hold_decision_cell_id=template.hold_decision_cell_id,
        classifier=classifier,
        strata=template.strata,
        efficacy_reserves=reserves,
        hold_reserve_anchor_slot_id=template.hold_reserve_anchor_slot_id,
        reserve_policy_id=template.reserve_policy_id,
        prospective_identity_domain_id=template.prospective_identity_domain_id,
        causal_cutoff_id=template.causal_cutoff_id,
        causal_cutoff=template.causal_cutoff,
        response_coordinates=template.response_coordinates,
        effect_quantity_id=template.effect_quantity_id,
        effect_native_unit=template.effect_native_unit,
        favorable_direction=template.favorable_direction,
        minimum_evaluable_efficacy_units=template.minimum_evaluable_efficacy_units,
        minimum_active_coverage=template.minimum_active_coverage,
        minimum_reference_correctness_coverage=(template.minimum_reference_correctness_coverage),
        alpha=template.alpha,
        degrees_of_freedom=template.degrees_of_freedom,
        one_sided_critical_value=template.one_sided_critical_value,
        materiality=template.materiality,
        intent_to_treat=template.intent_to_treat,
        reduction_order=template.reduction_order,
        mixed_efficacy_rule_id=template.mixed_efficacy_rule_id,
        all_positive_strengthening_rule_id=(template.all_positive_strengthening_rule_id),
        acquisition_domain_id=template.acquisition_domain_id,
        reference_domain_id=template.reference_domain_id,
        evaluation_domain_id=template.evaluation_domain_id,
        maximum_claim_ceiling=template.maximum_claim_ceiling,
        terminal_precedence=template.terminal_precedence,
        evidence_ceiling=EvidenceCeiling.CONTROLLER_USE,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    precommitment = MatchedActionHoldEvaluationPrecommitment(
        precommitment_id=template.expected_precommitment_id,
        authoring_bundle=authoring_bundle_identity,
        frozen_template=template_identity,
        selected_preparation_roster=selected_roster_identity,
        native_hold_selected_roster=native_roster_identity,
        eligible_admission_study=programme_identity,
        issued_manifest=issued_manifest,
        extension_set=extension_set,
        acquisition_plan_seal=acquisition_plan_seal_identity,
        matched_plan=plan,
        reference_design=reference_identity,
        evaluator_binding=template.evaluator_binding,
        logical_roster_rule_id=template.logical_roster_rule_id,
        causal_cutoff_id=template.causal_cutoff_id,
        acquisition_domain_id=template.acquisition_domain_id,
        reference_domain_id=template.reference_domain_id,
        evaluation_domain_id=template.evaluation_domain_id,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    receipt = MatchedActionHoldEvaluationTemplateBindingReceipt(
        receipt_id=receipt_id,
        authoring_bundle=authoring_bundle_identity,
        template=template_identity,
        selected_preparation_roster=selected_roster_identity,
        native_hold_selected_roster=native_roster_identity,
        native_hold_calibration=calibration_identity,
        measured_hold_fibre=measured_hold_identity,
        finite_chart_reference_design=reference_identity,
        classifier=classifier,
        eligible_admission_study=programme_identity,
        issued_manifest=issued_manifest,
        extension_set=extension_set,
        acquisition_plan_seal=acquisition_plan_seal_identity,
        evaluation_plan=ObjectIdentity.from_record(plan.evaluation_plan_id, plan),
        precommitment=ObjectIdentity.from_record(
            precommitment.precommitment_id,
            precommitment,
        ),
        binding_implementation_id=binding_implementation_id,
        binding_implementation_sha256=binding_implementation_sha256,
        outcome_dependent_choice=False,
        evidence_ceiling=EvidenceCeiling.ADMISSION,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    return plan, precommitment, receipt


class MatchedActionHoldControllerEvaluationTemplateBinder:
    "Code-owned deterministic wrapper for matched materialization after admission."

    def __init__(self, *, implementation_id: str, implementation_sha256: str) -> None:
        validate_stable_id(implementation_id, field_name="implementation_id")
        validate_sha256(implementation_sha256, field_name="implementation_sha256")
        self._implementation_id = implementation_id
        self._implementation_sha256 = implementation_sha256

    def bind(
        self,
        *,
        receipt_id: str,
        authoring_bundle: MatchedActionHoldProspectiveAuthoringBundle,
        selected_preparation_roster: MatchedActionHoldSelectedPreparationRoster,
        native_hold_selected_roster: NativeHoldCalibrationSelectedRoster,
        native_hold_calibration: NativeHoldDecisionCellCalibrationReceipt,
        measured_hold_fibre: AdmissionMeasuredHoldFibre,
        finite_chart_reference_design: FiniteChartReferenceDesign,
        classifier: ObjectIdentity,
        eligible_admission_study: AdmissionControllerStudy,
        issued_manifest: ObjectIdentity,
        extension_set: ObjectIdentity,
    ) -> tuple[
        MatchedActionHoldControllerEvaluationPlan,
        MatchedActionHoldEvaluationPrecommitment,
        MatchedActionHoldEvaluationTemplateBindingReceipt,
    ]:
        return bind_matched_action_hold_controller_evaluation_template(
            receipt_id=receipt_id,
            authoring_bundle=authoring_bundle,
            selected_preparation_roster=selected_preparation_roster,
            native_hold_selected_roster=native_hold_selected_roster,
            native_hold_calibration=native_hold_calibration,
            measured_hold_fibre=measured_hold_fibre,
            finite_chart_reference_design=finite_chart_reference_design,
            classifier=classifier,
            eligible_admission_study=eligible_admission_study,
            issued_manifest=issued_manifest,
            extension_set=extension_set,
            binding_implementation_id=self._implementation_id,
            binding_implementation_sha256=self._implementation_sha256,
        )


def decode_matched_action_hold_controller_evaluation_plan(
    payload: bytes,
) -> MatchedActionHoldControllerEvaluationPlan:
    return decode_canonical_bytes(
        payload,
        MatchedActionHoldControllerEvaluationPlan,
        maximum_bytes=MAX_MATCHED_ACTION_HOLD_PLAN_BYTES,
    )


def decode_matched_action_hold_acquisition_plan_seal(
    payload: bytes,
) -> MatchedActionHoldAcquisitionPlanSeal:
    return decode_canonical_bytes(
        payload,
        MatchedActionHoldAcquisitionPlanSeal,
        maximum_bytes=MAX_MATCHED_ACTION_HOLD_PLAN_BYTES,
    )


def decode_matched_action_hold_prospective_authoring_bundle(
    payload: bytes,
) -> MatchedActionHoldProspectiveAuthoringBundle:
    return decode_canonical_bytes(
        payload,
        MatchedActionHoldProspectiveAuthoringBundle,
        maximum_bytes=MAX_MATCHED_ACTION_HOLD_PLAN_BYTES,
    )


def decode_matched_action_hold_controller_evaluation_template(
    payload: bytes,
) -> MatchedActionHoldControllerEvaluationTemplate:
    return decode_canonical_bytes(
        payload,
        MatchedActionHoldControllerEvaluationTemplate,
        maximum_bytes=MAX_MATCHED_ACTION_HOLD_PLAN_BYTES,
    )


def decode_matched_action_hold_evaluation_precommitment(
    payload: bytes,
) -> MatchedActionHoldEvaluationPrecommitment:
    return decode_canonical_bytes(
        payload,
        MatchedActionHoldEvaluationPrecommitment,
        maximum_bytes=MAX_MATCHED_ACTION_HOLD_PLAN_BYTES,
    )


def decode_matched_action_hold_evaluation_template_binding_receipt(
    payload: bytes,
) -> MatchedActionHoldEvaluationTemplateBindingReceipt:
    return decode_canonical_bytes(
        payload,
        MatchedActionHoldEvaluationTemplateBindingReceipt,
        maximum_bytes=MAX_MATCHED_ACTION_HOLD_PLAN_BYTES,
    )


__all__ = [
    "MATCHED_ACTION_HOLD_ADJUDICATION_SCHEMA",
    "MATCHED_ACTION_HOLD_REDUCTION_ORDER",
    "MATCHED_ACTION_HOLD_REVEALED_BUNDLE_SCHEMA",
    "MATCHED_ACTION_HOLD_SEALED_BUNDLE_SCHEMA",
    "MATCHED_ACTION_HOLD_EVALUATION_TEMPLATE_SCHEMA",
    "MATCHED_ACTION_HOLD_OBSERVER_CONFIG_SCHEMA",
    "MATCHED_ACTION_HOLD_OUTCOME_PROJECTION_RECEIPT_SCHEMA",
    "MATCHED_ACTION_HOLD_SEALED_EPISODE_SCHEMA",
    "MATCHED_ACTION_HOLD_SELECTED_PREPARATION_ROSTER_SCHEMA",
    "MATCHED_ACTION_HOLD_TERMINAL_PRECEDENCE",
    "MAX_MATCHED_ACTION_HOLD_PLAN_BYTES",
    'MatchedActionHoldAcquisitionPlanSeal',
    'MatchedActionHoldControllerEvaluationTemplateBinder',
    'MatchedActionHoldControllerEvaluationTemplate',
    'MatchedActionHoldControllerEvaluationPlan',
    'MatchedActionHoldEfficacyReserveTemplate',
    'MatchedActionHoldEfficacyReserveSpec',
    'MatchedActionHoldEvaluationPrecommitment',
    'MatchedActionHoldEvaluationTemplateBindingReceipt',
    'MatchedActionHoldExecutionBranch',
    'MatchedActionHoldExecutionCellSpec',
    'MatchedActionHoldOutcomeProjectionEvidenceContract',
    'MatchedActionHoldPreparationMaterialization',
    'MatchedActionHoldProspectiveAuthoringBundle',
    'MatchedActionHoldSelectedPreparationRoster',
    'MatchedActionHoldUnitRole',
    'MatchedActionHoldUnitSpec',
    'MatchedActionHoldUnitTemplate',
    'bind_matched_action_hold_controller_evaluation_template',
    'build_matched_action_hold_acquisition_plan_seal',
    'build_matched_action_hold_prospective_authoring_bundle',
    'decode_matched_action_hold_acquisition_plan_seal',
    'decode_matched_action_hold_controller_evaluation_plan',
    'decode_matched_action_hold_controller_evaluation_template',
    'decode_matched_action_hold_evaluation_precommitment',
    'decode_matched_action_hold_evaluation_template_binding_receipt',
    'decode_matched_action_hold_prospective_authoring_bundle',
]
