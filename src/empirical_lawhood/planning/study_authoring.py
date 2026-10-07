"""Strict mutable-draft inputs for nonactuating candidate compilation.

The records in this module are an additive authoring boundary.  They reference
the accepted kernel and planning objects and never redefine their scientific
meaning.  A draft is mutable only as a file before issue; each decoded record
is an immutable value object.
"""

from __future__ import annotations

from empirical_lawhood.kernel.retrospective_prediction_experiments import RetrospectiveExperimentSpec

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import InformationCutoff

from .campaigns import CampaignSpec


class StudyDraftLifecycle(StrEnum):
    DRAFT = "DRAFT"


class DesignOriginKind(StrEnum):
    ORDINARY_THEORY_PREDECLARED = "ORDINARY_THEORY_PREDECLARED"
    PROSPECTIVE_NOMINATION = "PROSPECTIVE_NOMINATION"
    HISTORICAL_REANALYSIS = "HISTORICAL_REANALYSIS"


class DesignInputRole(StrEnum):
    READINESS_METADATA = "READINESS_METADATA"
    MOTIVATION = "MOTIVATION"
    DEVELOPMENT_TUNING = "DEVELOPMENT_TUNING"
    CLAIM_DERIVATION = "CLAIM_DERIVATION"


class SourceMaterializationRole(StrEnum):
    PREPARED_MEDIUM = "PREPARED_MEDIUM"
    OBSERVATION_STREAM = "OBSERVATION_STREAM"
    CALIBRATION = "CALIBRATION"
    RECEIVER_DEFINITION = "RECEIVER_DEFINITION"
    NUMERICAL_CONFIGURATION = "NUMERICAL_CONFIGURATION"


class SourceAccessDisposition(StrEnum):
    VERIFIED_ACCESS = "VERIFIED_ACCESS"
    ACQUISITION_AUTHORITY_REQUIRED = "ACQUISITION_AUTHORITY_REQUIRED"
    UNAVAILABLE = "UNAVAILABLE"


class ConditionalGateKind(StrEnum):
    ADMISSION_AND_REACHABILITY_PASSED = "ADMISSION_AND_REACHABILITY_PASSED"
    UPSTREAM_SCIENTIFIC_GATE_PASSED = "UPSTREAM_SCIENTIFIC_GATE_PASSED"


class ConditionalTerminalDisposition(StrEnum):
    EXECUTION_ELIGIBLE = "EXECUTION_ELIGIBLE"
    STOPPED_BY_FROZEN_SCIENTIFIC_GATE = "STOPPED_BY_FROZEN_SCIENTIFIC_GATE"
    NOT_ATTEMPTED_PREREQUISITE_NOT_MET = "NOT_ATTEMPTED_PREREQUISITE_NOT_MET"
    NOT_ATTEMPTED_AUTHORITY_REQUIRED = "NOT_ATTEMPTED_AUTHORITY_REQUIRED"
    UNEVALUABLE_WITH_TYPED_REASON = "UNEVALUABLE_WITH_TYPED_REASON"


@dataclass(frozen=True, slots=True)
class DesignOrigin(CanonicalRecord):
    """Exactly one ordinary or nomination-derived origin for a draft."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/design-origin'

    origin_id: str
    kind: DesignOriginKind
    declared_input_ids: tuple[str, ...]
    parent_visibility_ceiling: VisibilityCeiling
    nomination: ObjectIdentity | None = None
    requests_fresh_child: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.origin_id, field_name="origin_id")
        require_sorted_unique_strings(
            self.declared_input_ids,
            field_name="declared_input_ids",
            allow_empty=False,
        )
        if self.kind is DesignOriginKind.ORDINARY_THEORY_PREDECLARED:
            if self.nomination is not None:
                raise ValueError("ordinary design origin cannot bind a nomination")
            if self.parent_visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
                raise ValueError("ordinary design origin must begin with prospective visibility")
            if self.requests_fresh_child:
                raise ValueError("ordinary design origin is already the fresh design")
            return
        if self.kind is not DesignOriginKind.PROSPECTIVE_NOMINATION:
            raise ValueError("prospective origin cannot represent historical reanalysis")
        if self.nomination is None:
            raise ValueError("nomination-derived design origin requires a nomination identity")
        if self.parent_visibility_ceiling.is_promotable:
            raise ValueError("nomination-derived origin must retain nonpromotable visibility")


@dataclass(frozen=True, slots=True)
class RetrospectiveDesignOrigin(DesignOrigin):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/retrospective-design-origin'

    def __post_init__(self) -> None:
        validate_stable_id(self.origin_id, field_name="origin_id")
        require_sorted_unique_strings(
            self.declared_input_ids, field_name="declared_input_ids", allow_empty=False
        )
        if self.kind is not DesignOriginKind.HISTORICAL_REANALYSIS:
            raise ValueError("historical design requires historical reanalysis origin")
        if self.parent_visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE:
            raise ValueError("historical design must retain outcome exposure")
        if self.nomination is not None or self.requests_fresh_child:
            raise ValueError("historical reanalysis does not nominate or grant fresh evidence")


@dataclass(frozen=True, slots=True)
class DesignInputRecord(CanonicalRecord):
    """One exact input and its permanent role in design visibility."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/design-input-record'

    input_id: str
    object_identity: ObjectIdentity
    materialization_sha256: str
    information_cutoff: InformationCutoff
    role: DesignInputRole
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    operator_id: str
    physical_unit_ids: tuple[str, ...] = ()
    seed_ids: tuple[str, ...] = ()
    parent_input_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        validate_sha256(
            self.materialization_sha256,
            field_name="materialization_sha256",
        )
        validate_stable_id(self.operator_id, field_name="operator_id")
        for name, values in (
            ("physical_unit_ids", self.physical_unit_ids),
            ("seed_ids", self.seed_ids),
            ("parent_input_ids", self.parent_input_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if self.input_id in self.parent_input_ids:
            raise ValueError("design input cannot parent itself")


@dataclass(frozen=True, slots=True)
class UnresolvedDecision(CanonicalRecord):
    """A closed, explicit material decision that prevents candidate issue."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/unresolved-decision'

    decision_id: str
    field_path: str
    question: str
    option_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        validate_nonempty(self.field_path, field_name="field_path")
        validate_nonempty(self.question, field_name="question")
        require_sorted_unique_strings(
            self.option_ids,
            field_name="option_ids",
            allow_empty=False,
        )
        if len(self.option_ids) < 2:
            raise ValueError("unresolved decision requires at least two options")


@dataclass(frozen=True, slots=True)
class CapabilitySelection(CanonicalRecord):
    """Static capability identity selected by a draft."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/capability-selection'

    capability_key: str
    capability_version: str
    implementation_sha256: str

    @property
    def selection_id(self) -> str:
        return f"{self.capability_key}@{self.capability_version}"

    def __post_init__(self) -> None:
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_sha256(
            self.implementation_sha256,
            field_name="implementation_sha256",
        )


@dataclass(frozen=True, slots=True)
class SourceMaterializationRef(CanonicalRecord):
    """Content-addressed source candidate; never a path, query or command."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/source-materialization-ref'

    source_id: str
    role: SourceMaterializationRole
    evidence_world_id: str
    materialization: ObjectIdentity
    content_sha256: str
    source_config_sha256: str
    observation_operator: ObjectIdentity
    numerical_view_ids: tuple[str, ...]
    qualification_receipt: ObjectIdentity | None
    access_disposition: SourceAccessDisposition

    def __post_init__(self) -> None:
        validate_stable_id(self.source_id, field_name="source_id")
        validate_stable_id(self.evidence_world_id, field_name="evidence_world_id")
        validate_sha256(self.content_sha256, field_name="content_sha256")
        validate_sha256(
            self.source_config_sha256,
            field_name="source_config_sha256",
        )
        require_sorted_unique_strings(
            self.numerical_view_ids,
            field_name="numerical_view_ids",
            allow_empty=False,
        )
        if (
            self.access_disposition is SourceAccessDisposition.VERIFIED_ACCESS
            and self.qualification_receipt is None
        ):
            raise ValueError("verified source access requires a qualification receipt")


@dataclass(frozen=True, slots=True)
class MaterializationQualificationReceipt(CanonicalRecord):
    """Experiment-specific semantic qualification of exact source bytes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/materialization-qualification-receipt'

    receipt_id: str
    source_id: str
    materialization: ObjectIdentity
    content_sha256: str
    evidence_world_id: str
    observation_operator: ObjectIdentity
    numerical_view_ids: tuple[str, ...]
    native_unit_ids: tuple[str, ...]
    frame_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]
    receiver_semantics_id: str
    validity_contract_id: str
    uncertainty_contract_id: str
    access_disposition: SourceAccessDisposition
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("source_id", self.source_id),
            ("evidence_world_id", self.evidence_world_id),
            ("receiver_semantics_id", self.receiver_semantics_id),
            ("validity_contract_id", self.validity_contract_id),
            ("uncertainty_contract_id", self.uncertainty_contract_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(self.content_sha256, field_name="content_sha256")
        for name, values in (
            ("numerical_view_ids", self.numerical_view_ids),
            ("native_unit_ids", self.native_unit_ids),
            ("frame_ids", self.frame_ids),
            ("clock_ids", self.clock_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if self.access_disposition is not SourceAccessDisposition.VERIFIED_ACCESS:
            raise ValueError("qualification receipt requires verified access disposition")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_SEALED,
        }:
            raise ValueError("source qualification cannot reveal evaluation outcomes")


@dataclass(frozen=True, slots=True)
class ConditionalChildRequest(CanonicalRecord):
    "Pre-outcome request for a completely frozen conditional child."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/conditional-child-request'

    request_id: str
    parent_node_id: str
    parent_receipt_input_id: str
    gate_kind: ConditionalGateKind
    template_key: str
    evaluation_unit_ids: tuple[str, ...]
    eligible_disposition: ConditionalTerminalDisposition
    ineligible_disposition: ConditionalTerminalDisposition

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("parent_node_id", self.parent_node_id),
            ("parent_receipt_input_id", self.parent_receipt_input_id),
            ("template_key", self.template_key),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.evaluation_unit_ids,
            field_name="evaluation_unit_ids",
            allow_empty=False,
        )
        if self.eligible_disposition is not ConditionalTerminalDisposition.EXECUTION_ELIGIBLE:
            raise ValueError("eligible conditional branch must be execution eligible")
        if self.ineligible_disposition is ConditionalTerminalDisposition.EXECUTION_ELIGIBLE:
            raise ValueError("ineligible conditional branch must terminate or stop")


@dataclass(frozen=True, slots=True)
class StudyDraft(CanonicalRecord):
    """One mutable-on-disk strict draft decoded into immutable typed values."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/study-draft'

    draft_id: str
    lifecycle: StudyDraftLifecycle
    question: str
    alternative_ids: tuple[str, ...]
    design_origin: DesignOrigin
    design_inputs: tuple[DesignInputRecord, ...]
    development_unit_ids: tuple[str, ...]
    evaluation_unit_ids: tuple[str, ...]
    development_seed_ids: tuple[str, ...]
    evaluation_seed_ids: tuple[str, ...]
    unresolved_decisions: tuple[UnresolvedDecision, ...]
    system: SystemSpec | None
    experiment: ExperimentSpec | None
    campaign: CampaignSpec | None
    dag_template_key: str
    capability_selections: tuple[CapabilitySelection, ...]
    source_materializations: tuple[SourceMaterializationRef, ...]
    resource_ceiling: ResourceBudget
    conditional_successor: ConditionalChildRequest | None = None

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveStudyDraft):
            if type(self.design_origin) is not DesignOrigin:
                raise ValueError("ProgrammeDraft requires its original design_origin schema")
            if self.experiment is not None and type(self.experiment) is not ExperimentSpec:
                raise ValueError("ProgrammeDraft requires its original experiment schema")
        validate_stable_id(self.draft_id, field_name="draft_id")
        if self.lifecycle is not StudyDraftLifecycle.DRAFT:
            raise ValueError("authoring root must remain in DRAFT lifecycle")
        validate_nonempty(self.question, field_name="question")
        require_sorted_unique_strings(
            self.alternative_ids,
            field_name="alternative_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.design_inputs,
            attribute="input_id",
            field_name="design_inputs",
        )
        for name, values in (
            ("development_unit_ids", self.development_unit_ids),
            ("evaluation_unit_ids", self.evaluation_unit_ids),
            ("development_seed_ids", self.development_seed_ids),
            ("evaluation_seed_ids", self.evaluation_seed_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        require_sorted_unique_ids(
            self.unresolved_decisions,
            attribute="decision_id",
            field_name="unresolved_decisions",
        )
        validate_stable_id(self.dag_template_key, field_name="dag_template_key")
        require_sorted_unique_ids(
            self.capability_selections,
            attribute="selection_id",
            field_name="capability_selections",
        )
        require_sorted_unique_ids(
            self.source_materializations,
            attribute="source_id",
            field_name="source_materializations",
        )


__all__ = [
    "CapabilitySelection",
    "ConditionalGateKind",
    'ConditionalChildRequest',
    "ConditionalTerminalDisposition",
    "DesignInputRecord",
    "DesignInputRole",
    "DesignOrigin",
    "DesignOriginKind",
    "MaterializationQualificationReceipt",
    'StudyDraft',
    'StudyDraftLifecycle',
    "SourceAccessDisposition",
    "SourceMaterializationRef",
    "SourceMaterializationRole",
    "UnresolvedDecision",
]


@dataclass(frozen=True, slots=True)
class RetrospectiveStudyDraft(StudyDraft):
    "Explicit retrospective authoring compatibility under its declared schema."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/retrospective-study-draft'

    design_origin: RetrospectiveDesignOrigin
    experiment: RetrospectiveExperimentSpec | None

    def __post_init__(self) -> None:
        if type(self.design_origin) is not RetrospectiveDesignOrigin:
            raise ValueError("RetrospectiveStudyDraft requires its exact design_origin schema")
        if self.experiment is not None and type(self.experiment) is not RetrospectiveExperimentSpec:
            raise ValueError("RetrospectiveStudyDraft requires its exact experiment schema")
        StudyDraft.__post_init__(self)
        if self.conditional_successor is not None:
            raise ValueError("historical reanalysis cannot issue a prospective conditional child")
        if self.development_unit_ids or self.evaluation_unit_ids:
            raise ValueError("historical split records cannot be counted as physical units")
        if self.development_seed_ids or self.evaluation_seed_ids:
            raise ValueError("historical reanalysis does not acquire new simulation seeds")
