"""Typed, noncompensating evidence fan-in for physical scale morphism adjudication.

The records in this module do not adjudicate a claim.  They make the compact
IP-12 inputs explicit enough that :mod:`adjudication` can derive its facts
instead of accepting a manually populated collection of booleans.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.reference_worlds.physical_scale_morphism.contracts import PHYSICAL_SCALE_MORPHISM_TRUTH_FIXTURE_IDS, PhysicalScaleMorphismTruthMethodSuiteResult

from .comparators import PhysicalScaleMorphismCoordinateComparatorPanel
from .contracts import PhysicalScaleMorphismDefectPanel, PhysicalScaleMorphismForecastTripletScore, PhysicalScaleMorphismHoldFibreRecord, PhysicalScaleMorphismMapFamily, PhysicalScaleMorphismObstructionSet, PhysicalScaleMorphismSemanticRolePanel
from .defects import PhysicalScaleMorphismDefectThresholds
from .governance import PhysicalScaleMorphismConstructQualification, PhysicalScaleMorphismCompleteFanIn, PhysicalScaleMorphismDevelopmentHandoff, PhysicalScaleMorphismExecutionSeal, PhysicalScaleMorphismImplementationRecurrencePanel, PhysicalScaleMorphismIssueRecord, PhysicalScaleMorphismIssueTerminal, PhysicalScaleMorphismPredictionFreeze, PhysicalScaleMorphismSourceApparatusQualification
from .heterogeneity import PhysicalScaleMorphismHeterogeneityPanel
from .morphisms import PhysicalScaleMorphismCompositionPanel


class PhysicalScaleMorphismEvidenceState(StrEnum):
    OPPOSED = "OPPOSED"
    SUPPORTED = "SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"


class PhysicalScaleMorphismSectionKind(StrEnum):
    BOUNDARY = "BOUNDARY"
    CATEGORICAL = "CATEGORICAL"
    DYNAMICAL = "DYNAMICAL"
    METRIC = "METRIC"
    SCALE_FLOW = "SCALE_FLOW"


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismMorphismAxisEvidence(CanonicalRecord):
    """Exact map roster and its noncompensating evaluation disposition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-morphism-axis-evidence'

    panel_id: str
    map_family: PhysicalScaleMorphismMapFamily
    required_morphism_ids: tuple[str, ...]
    supported_morphism_ids: tuple[str, ...]
    opposed_morphism_ids: tuple[str, ...]
    unevaluable_morphism_ids: tuple[str, ...]
    required_negative_control_ids: tuple[str, ...]
    rejected_negative_control_ids: tuple[str, ...]
    false_safe_cell_ids: tuple[str, ...]
    crossover_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        for name in (
            "required_morphism_ids",
            "supported_morphism_ids",
            "opposed_morphism_ids",
            "unevaluable_morphism_ids",
            "required_negative_control_ids",
            "rejected_negative_control_ids",
            "false_safe_cell_ids",
            "crossover_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name not in {"required_morphism_ids", "required_negative_control_ids"},
            )
        dispositions = (
            set(self.supported_morphism_ids),
            set(self.opposed_morphism_ids),
            set(self.unevaluable_morphism_ids),
        )
        if any(
            left & right
            for index, left in enumerate(dispositions)
            for right in dispositions[index + 1 :]
        ):
            raise ValueError("one morphism cannot have two evaluation dispositions")
        if set().union(*dispositions) != set(self.required_morphism_ids):
            raise ValueError("morphism-axis dispositions do not cover the exact frozen roster")
        if not set(self.rejected_negative_control_ids).issubset(self.required_negative_control_ids):
            raise ValueError("axis rejected an unregistered negative control")
        if self.crossover_ids and self.map_family is not PhysicalScaleMorphismMapFamily.PHYSICAL_SCALE:
            raise ValueError("physical crossover can occur only on the physical-scale axis")

    @property
    def state(self) -> PhysicalScaleMorphismEvidenceState:
        if self.unevaluable_morphism_ids:
            return PhysicalScaleMorphismEvidenceState.UNEVALUABLE
        if (
            not self.opposed_morphism_ids
            and not self.false_safe_cell_ids
            and not self.crossover_ids
            and self.rejected_negative_control_ids == self.required_negative_control_ids
        ):
            return PhysicalScaleMorphismEvidenceState.SUPPORTED
        return PhysicalScaleMorphismEvidenceState.OPPOSED


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismSectionEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-section-evidence'

    panel_id: str
    section: PhysicalScaleMorphismSectionKind
    required_property_ids: tuple[str, ...]
    surviving_property_ids: tuple[str, ...]
    opposed_property_ids: tuple[str, ...]
    unevaluable_property_ids: tuple[str, ...]
    boundary_complex_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        for name in (
            "required_property_ids",
            "surviving_property_ids",
            "opposed_property_ids",
            "unevaluable_property_ids",
            "boundary_complex_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name != "required_property_ids",
            )
        dispositions = (
            set(self.surviving_property_ids),
            set(self.opposed_property_ids),
            set(self.unevaluable_property_ids),
        )
        if any(
            left & right
            for index, left in enumerate(dispositions)
            for right in dispositions[index + 1 :]
        ):
            raise ValueError("one section property cannot have two dispositions")
        if set().union(*dispositions) != set(self.required_property_ids):
            raise ValueError("section dispositions do not cover the exact property roster")
        if self.section is PhysicalScaleMorphismSectionKind.BOUNDARY:
            if not self.boundary_complex_ids:
                raise ValueError("boundary section must bind its finite boundary complexes")
        elif self.boundary_complex_ids:
            raise ValueError("only boundary section evidence may bind boundary complexes")

    @property
    def state(self) -> PhysicalScaleMorphismEvidenceState:
        if self.unevaluable_property_ids:
            return PhysicalScaleMorphismEvidenceState.UNEVALUABLE
        if not self.opposed_property_ids:
            return PhysicalScaleMorphismEvidenceState.SUPPORTED
        return PhysicalScaleMorphismEvidenceState.OPPOSED


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismNativeQuantityControl(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-native-quantity-control'

    panel_id: str
    required_nonfixed_quantity_ids: tuple[str, ...]
    observed_nonfixed_quantity_ids: tuple[str, ...]
    fixed_against_prediction_ids: tuple[str, ...]
    unevaluable_quantity_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        for name in (
            "required_nonfixed_quantity_ids",
            "observed_nonfixed_quantity_ids",
            "fixed_against_prediction_ids",
            "unevaluable_quantity_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name != "required_nonfixed_quantity_ids",
            )
        dispositions = (
            set(self.observed_nonfixed_quantity_ids),
            set(self.fixed_against_prediction_ids),
            set(self.unevaluable_quantity_ids),
        )
        if any(
            left & right
            for index, left in enumerate(dispositions)
            for right in dispositions[index + 1 :]
        ):
            raise ValueError("native quantity control dispositions overlap")
        if set().union(*dispositions) != set(self.required_nonfixed_quantity_ids):
            raise ValueError("native quantity controls do not cover the frozen roster")

    @property
    def state(self) -> PhysicalScaleMorphismEvidenceState:
        if self.unevaluable_quantity_ids:
            return PhysicalScaleMorphismEvidenceState.UNEVALUABLE
        if not self.fixed_against_prediction_ids:
            return PhysicalScaleMorphismEvidenceState.SUPPORTED
        return PhysicalScaleMorphismEvidenceState.OPPOSED


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismNegativeControlAudit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-negative-control-audit'

    audit_id: str
    required_control_ids: tuple[str, ...]
    rejected_control_ids: tuple[str, ...]
    failed_control_ids: tuple[str, ...]
    unevaluable_control_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        for name in (
            "required_control_ids",
            "rejected_control_ids",
            "failed_control_ids",
            "unevaluable_control_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name != "required_control_ids",
            )
        dispositions = (
            set(self.rejected_control_ids),
            set(self.failed_control_ids),
            set(self.unevaluable_control_ids),
        )
        if any(
            left & right
            for index, left in enumerate(dispositions)
            for right in dispositions[index + 1 :]
        ):
            raise ValueError("negative-control dispositions overlap")
        if set().union(*dispositions) != set(self.required_control_ids):
            raise ValueError("negative-control audit does not cover the frozen roster")

    @property
    def all_rejected(self) -> bool:
        return self.rejected_control_ids == self.required_control_ids


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismMultiplicityAudit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-multiplicity-audit'

    audit_id: str
    frozen_family_ids: tuple[str, ...]
    applied_family_ids: tuple[str, ...]
    board_resampling_family_ids: tuple[str, ...]
    row_or_view_resampling_family_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        for name in (
            "frozen_family_ids",
            "applied_family_ids",
            "board_resampling_family_ids",
            "row_or_view_resampling_family_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name not in {"frozen_family_ids", "applied_family_ids"},
            )
        if not set(self.board_resampling_family_ids).issubset(self.applied_family_ids):
            raise ValueError("board-resampling family lies outside applied multiplicity families")
        if not set(self.row_or_view_resampling_family_ids).issubset(self.applied_family_ids):
            raise ValueError("nested-row/view resampling family lies outside applied families")

    @property
    def violation(self) -> bool:
        return (
            self.frozen_family_ids != self.applied_family_ids
            or set(self.board_resampling_family_ids) != set(self.applied_family_ids)
            or bool(self.row_or_view_resampling_family_ids)
        )


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismAuthorityIndependenceAudit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-authority-independence-audit'

    audit_id: str
    required_authority_ids: tuple[str, ...]
    presented_authority_ids: tuple[str, ...]
    prohibited_access_event_ids: tuple[str, ...]
    incompatible_role_overlap_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        for name in (
            "required_authority_ids",
            "presented_authority_ids",
            "prohibited_access_event_ids",
            "incompatible_role_overlap_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name not in {"required_authority_ids", "presented_authority_ids"},
            )

    @property
    def violation(self) -> bool:
        return (
            not set(self.required_authority_ids).issubset(self.presented_authority_ids)
            or bool(self.prohibited_access_event_ids)
            or bool(self.incompatible_role_overlap_ids)
        )


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismBoardBatchBinding(CanonicalRecord):
    """Compact decoded linkage from one issued board to scale/batch/panel."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-board-batch-binding'

    binding_id: str
    board_id: str
    scale_cells: int
    batch_id: str
    panel_id: str
    board_identity: ObjectIdentity
    metrology_identity: ObjectIdentity

    def __post_init__(self) -> None:
        for name in ("binding_id", "board_id", "batch_id", "panel_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.scale_cells not in {16, 32, 64}:
            raise ValueError("board/batch binding lies outside the physical scale roster")
        if (
            self.board_identity.object_schema != 'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-board-identity'
            or self.board_identity.object_id != self.board_id
        ):
            raise ValueError("board/batch binding contains the wrong board identity")
        if self.metrology_identity.object_schema != 'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-component-metrology':
            raise ValueError("board/batch binding contains the wrong metrology identity")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismAdjudicationEvidenceBundle(CanonicalRecord):
    """Complete compact evaluator input from which adjudication facts derive."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-adjudication-evidence-bundle'

    bundle_id: str
    method_qualification: PhysicalScaleMorphismTruthMethodSuiteResult
    construct: PhysicalScaleMorphismConstructQualification
    source_apparatus: PhysicalScaleMorphismSourceApparatusQualification
    development: PhysicalScaleMorphismDevelopmentHandoff
    prediction_freeze: PhysicalScaleMorphismPredictionFreeze
    issue: PhysicalScaleMorphismIssueRecord
    execution: PhysicalScaleMorphismExecutionSeal
    fan_in: PhysicalScaleMorphismCompleteFanIn
    board_batch_bindings: tuple[PhysicalScaleMorphismBoardBatchBinding, ...]
    coordinate_panel: PhysicalScaleMorphismCoordinateComparatorPanel
    forecast_scores: tuple[PhysicalScaleMorphismForecastTripletScore, ...]
    defect_thresholds: PhysicalScaleMorphismDefectThresholds
    defect_panels: tuple[PhysicalScaleMorphismDefectPanel, ...]
    semantic_panels: tuple[PhysicalScaleMorphismSemanticRolePanel, ...]
    required_hold_ids: tuple[str, ...]
    hold_records: tuple[PhysicalScaleMorphismHoldFibreRecord, ...]
    morphism_axes: tuple[PhysicalScaleMorphismMorphismAxisEvidence, ...]
    composition_panels: tuple[PhysicalScaleMorphismCompositionPanel, ...]
    section_panels: tuple[PhysicalScaleMorphismSectionEvidence, ...]
    native_quantity_control: PhysicalScaleMorphismNativeQuantityControl
    heterogeneity_panels: tuple[PhysicalScaleMorphismHeterogeneityPanel, ...]
    obstruction_sets: tuple[PhysicalScaleMorphismObstructionSet, ...]
    negative_control_audit: PhysicalScaleMorphismNegativeControlAudit
    multiplicity_audit: PhysicalScaleMorphismMultiplicityAudit
    authority_independence_audit: PhysicalScaleMorphismAuthorityIndependenceAudit
    implementation_recurrence: PhysicalScaleMorphismImplementationRecurrencePanel

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        require_sorted_unique_ids(
            self.board_batch_bindings,
            attribute="binding_id",
            field_name="board_batch_bindings",
        )
        require_sorted_unique_ids(
            self.forecast_scores, attribute="score_id", field_name="forecast_scores"
        )
        require_sorted_unique_ids(
            self.defect_panels, attribute="panel_id", field_name="defect_panels"
        )
        require_sorted_unique_ids(
            self.semantic_panels, attribute="panel_id", field_name="semantic_panels"
        )
        require_sorted_unique_strings(
            self.required_hold_ids, field_name="required_hold_ids", allow_empty=False
        )
        require_sorted_unique_ids(self.hold_records, attribute="hold_id", field_name="hold_records")
        require_sorted_unique_ids(
            self.morphism_axes, attribute="panel_id", field_name="morphism_axes"
        )
        require_sorted_unique_ids(
            self.composition_panels,
            attribute="panel_id",
            field_name="composition_panels",
        )
        require_sorted_unique_ids(
            self.section_panels, attribute="panel_id", field_name="section_panels"
        )
        require_sorted_unique_ids(
            self.heterogeneity_panels,
            attribute="panel_id",
            field_name="heterogeneity_panels",
        )
        require_sorted_unique_ids(
            self.obstruction_sets,
            attribute="obstruction_id",
            field_name="obstruction_sets",
        )
        required_nonempty = (
            self.forecast_scores,
            self.defect_panels,
            self.semantic_panels,
            self.hold_records,
            self.morphism_axes,
            self.composition_panels,
            self.heterogeneity_panels,
            self.obstruction_sets,
        )
        if any(not value for value in required_nonempty):
            raise ValueError("adjudication evidence bundle omits a required panel family")
        if (
            not self.method_qualification.method_qualified
            or self.method_qualification.passed_case_ids != PHYSICAL_SCALE_MORPHISM_TRUTH_FIXTURE_IDS
        ):
            raise ValueError("adjudication evidence lacks complete truth-method qualification")
        construct_identity = ObjectIdentity.from_record(
            self.construct.qualification_id, self.construct
        )
        source_identity = ObjectIdentity.from_record(
            self.source_apparatus.qualification_id, self.source_apparatus
        )
        development_identity = ObjectIdentity.from_record(
            self.development.handoff_id, self.development
        )
        if self.source_apparatus.construct_qualification != construct_identity:
            raise ValueError("source/apparatus evidence binds another construct qualification")
        if self.development.source_apparatus_qualification != source_identity:
            raise ValueError("development evidence binds another source/apparatus qualification")
        if self.prediction_freeze.construct_qualification != construct_identity:
            raise ValueError("prediction freeze binds another construct qualification")
        if self.prediction_freeze.source_apparatus_qualification != source_identity:
            raise ValueError("prediction freeze binds another source/apparatus qualification")
        if self.prediction_freeze.development_handoff != development_identity:
            raise ValueError("prediction freeze binds another development handoff")
        freeze_identity = ObjectIdentity.from_record(
            self.prediction_freeze.freeze_id, self.prediction_freeze
        )
        issue_identity = ObjectIdentity.from_record(self.issue.issue_id, self.issue)
        execution_identity = ObjectIdentity.from_record(self.execution.seal_id, self.execution)
        if (
            self.issue.prediction_freeze != freeze_identity
            or self.issue.issued_candidate != self.prediction_freeze.candidate
            or self.issue.terminal is not PhysicalScaleMorphismIssueTerminal.ISSUED_COMPLETE_ROSTER
        ):
            raise ValueError("adjudication evidence does not bind a complete frozen issue")
        if (
            self.execution.prediction_freeze != freeze_identity
            or self.execution.issue_record != issue_identity
            or self.execution.run != self.issue.issued_run
            or not self.execution.complete
        ):
            raise ValueError("adjudication evidence does not bind a complete execution seal")
        if (
            self.fan_in.prediction_freeze != freeze_identity
            or self.fan_in.execution_seal != execution_identity
            or not self.fan_in.complete
        ):
            raise ValueError("adjudication evidence does not bind a complete frozen fan-in")
        if (
            self.fan_in.expected_unit_ids != self.execution.expected_unit_ids
            or self.fan_in.received_unit_ids != self.execution.sealed_bundle_unit_ids
            or self.fan_in.terminal_unit_ids != self.execution.typed_terminal_unit_ids
        ):
            raise ValueError("adjudication fan-in roster differs from sealed execution")
        if not set(self.execution.publication_receipt_ids).issubset(
            self.fan_in.publication_receipt_ids
        ):
            raise ValueError("adjudication fan-in omits execution publication receipts")
        expected_sealed_artifacts = {
            *self.execution.physical_bundle_artifact_ids,
            *self.execution.numerical_artifact_ids,
        }
        if set(self.fan_in.sealed_payload_artifact_ids) != expected_sealed_artifacts:
            raise ValueError("adjudication fan-in payload differs from sealed execution artifacts")
        binding_by_board = {value.board_id: value for value in self.board_batch_bindings}
        if len(binding_by_board) != len(self.board_batch_bindings):
            raise ValueError("board/batch bindings repeat a board")
        if set(binding_by_board) != set(self.prediction_freeze.evaluation_board_ids):
            raise ValueError("board/batch bindings do not cover the evaluation roster")
        if {value.board_identity for value in self.board_batch_bindings} != set(
            self.issue.board_identity_records
        ):
            raise ValueError("board/batch bindings differ from issued board identities")
        if {value.metrology_identity for value in self.board_batch_bindings} != set(
            self.issue.pre_response_metrology_records
        ):
            raise ValueError("board/batch bindings differ from issued metrology identities")
        if {
            ObjectIdentity.from_record(value.binding_id, value)
            for value in self.board_batch_bindings
        } != set(self.issue.board_batch_binding_records):
            raise ValueError("board/batch bindings differ from immutable issued linkage records")
        for roster in self.prediction_freeze.evaluation_scale_rosters:
            bindings = tuple(
                value
                for value in self.board_batch_bindings
                if value.scale_cells == roster.scale_cells
            )
            if {value.board_id for value in bindings} != set(roster.board_ids):
                raise ValueError("board/batch scale linkage differs from the prediction freeze")
            if {value.batch_id for value in bindings} != set(roster.batch_ids):
                raise ValueError("actual batch linkage differs from the frozen batch roster")
        frozen_by_schema: dict[str, set[str]] = {}
        for value in self.prediction_freeze.frozen_object_identities:
            frozen_by_schema.setdefault(value.object_schema, set()).add(value.object_id)
        if self.coordinate_panel.selected_encoding_id not in frozen_by_schema.get(
            'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-comparator-encoding', set()
        ):
            raise ValueError("coordinate panel does not score the frozen selected encoding")
        frozen_triplet_identities = {
            value
            for value in self.prediction_freeze.frozen_object_identities
            if value.object_schema == 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-forecast-triplet'
        }
        if {value.triplet for value in self.forecast_scores} != frozen_triplet_identities:
            raise ValueError("forecast scores do not cover the exact frozen triplet roster")
        frozen_morphisms = frozen_by_schema.get('empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-morphism-record', set())
        axis_morphisms = [
            morphism_id
            for value in self.morphism_axes
            for morphism_id in value.required_morphism_ids
        ]
        if len(set(axis_morphisms)) != len(axis_morphisms) or set(axis_morphisms) != (
            frozen_morphisms
        ):
            raise ValueError("morphism axes do not partition the exact frozen map roster")
        if {value.map_family for value in self.morphism_axes} != set(PhysicalScaleMorphismMapFamily):
            raise ValueError("adjudication evidence lacks a frozen map-family axis")
        if {value.section for value in self.section_panels} != set(PhysicalScaleMorphismSectionKind):
            raise ValueError("adjudication evidence lacks a fixed-section/scale-flow panel")
        if {value.hold_id for value in self.hold_records} != set(self.required_hold_ids):
            raise ValueError("hold records do not cover the exact frozen hold roster")
        evaluation_boards = set(self.prediction_freeze.evaluation_board_ids)
        if {value.board_id for value in self.hold_records} != evaluation_boards:
            raise ValueError("hold evidence does not cover every frozen evaluation board")
        heterogeneity_boards = {
            result.board_id for panel in self.heterogeneity_panels for result in panel.board_results
        }
        if heterogeneity_boards != evaluation_boards:
            raise ValueError("heterogeneity evidence does not cover every evaluation board")
        if any(
            result.batch_id != binding_by_board[result.board_id].batch_id
            for panel in self.heterogeneity_panels
            for result in panel.board_results
        ):
            raise ValueError("heterogeneity batch labels differ from issued board bindings")
        frozen_contracts = frozen_by_schema.get(
            'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-morphism-composition-contract', set()
        )
        if {
            value.composition_contract.contract_id for value in self.composition_panels
        } != frozen_contracts:
            raise ValueError("composition panels do not cover the frozen contracts")
        if self.defect_thresholds.thresholds_id not in frozen_by_schema.get(
            'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-defect-thresholds', set()
        ):
            raise ValueError("adjudication uses thresholds absent from the prediction freeze")
        if self.authority_independence_audit.required_authority_ids != (
            self.prediction_freeze.authority_requirement_ids
        ):
            raise ValueError("authority audit differs from the frozen authority roster")


__all__ = [
    'PhysicalScaleMorphismAdjudicationEvidenceBundle',
    'PhysicalScaleMorphismAuthorityIndependenceAudit',
    'PhysicalScaleMorphismBoardBatchBinding',
    'PhysicalScaleMorphismEvidenceState',
    'PhysicalScaleMorphismMorphismAxisEvidence',
    'PhysicalScaleMorphismMultiplicityAudit',
    'PhysicalScaleMorphismNativeQuantityControl',
    'PhysicalScaleMorphismNegativeControlAudit',
    'PhysicalScaleMorphismSectionEvidence',
    'PhysicalScaleMorphismSectionKind',
]
