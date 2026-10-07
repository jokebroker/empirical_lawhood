"Shared structural-metatheory vocabulary with no scientific authority.\n\nThese values form a ladder parallel to ordinary measurement through controller-use evidence.  They do not\npromote a response law, admission result, controller or prospective controller-use result.\n"

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)


class MetatheoryClaimKind(StrEnum):
    SOURCE_PREPARATION_QUALIFICATION = "SOURCE_PREPARATION_QUALIFICATION"
    COORDINATE_CONSTRUCT_VALIDITY = "COORDINATE_CONSTRUCT_VALIDITY"
    RECEIVER_HISTORY_CLOSURE = "RECEIVER_HISTORY_CLOSURE"
    CHART_QUALIFICATION = "CHART_QUALIFICATION"
    DECISION_ASSURANCE = "DECISION_ASSURANCE"
    PROPERTY_TRANSPORT = "PROPERTY_TRANSPORT"
    STRUCTURAL_RECURRENCE = "STRUCTURAL_RECURRENCE"


class MetatheoryPredictiveLevel(StrEnum):
    CATEGORICAL = "CATEGORICAL"
    METRIC = "METRIC"
    DYNAMICAL = "DYNAMICAL"


class MetatheoryEvidenceCeiling(StrEnum):
    CONTRACT_CONFORMANCE = "CONTRACT_CONFORMANCE"
    DEVELOPMENT_STRUCTURAL = "DEVELOPMENT_STRUCTURAL"
    PROSPECTIVE_TARGET_LOCAL = "PROSPECTIVE_TARGET_LOCAL"


class MetatheoryCellDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    UNEVALUABLE = "UNEVALUABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class MetatheoryAggregateDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    MIXED = "MIXED"
    UNEVALUABLE = "UNEVALUABLE"
    PREREQUISITE_NONATTEMPT = "PREREQUISITE_NONATTEMPT"


class EvidenceDependenceClass(StrEnum):
    SAME_IMPLEMENTATION_RESAMPLE = "SAME_IMPLEMENTATION_RESAMPLE"
    INDEPENDENT_GENERATOR_SHARED_ONTOLOGY = "INDEPENDENT_GENERATOR_SHARED_ONTOLOGY"
    INDEPENDENT_IMPLEMENTATION_SHARED_SUBSTRATE = "INDEPENDENT_IMPLEMENTATION_SHARED_SUBSTRATE"
    INDEPENDENT_SUBSTRATE = "INDEPENDENT_SUBSTRATE"
    PHYSICAL_GROUNDING = "PHYSICAL_GROUNDING"


class MetatheoryObstructionKind(StrEnum):
    OBSERVED_OPPOSITION = "OBSERVED_OPPOSITION"
    OPERAND_ABSENT = "OPERAND_ABSENT"
    SOURCE_OR_DENOMINATOR_FAILURE = "SOURCE_OR_DENOMINATOR_FAILURE"
    COORDINATE_UNTARGETABLE = "COORDINATE_UNTARGETABLE"
    ACTION_CHAIN_FAILURE = "ACTION_CHAIN_FAILURE"
    DECISION_ASSURANCE_FAILURE = "DECISION_ASSURANCE_FAILURE"
    INSUFFICIENT_POWER = "INSUFFICIENT_POWER"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    PREREQUISITE_NONATTEMPT = "PREREQUISITE_NONATTEMPT"
    UNRESOLVED = "UNRESOLVED"


class LegitimateNextAct(StrEnum):
    """Non-authorizing bounded advice attached to an obstruction."""

    CONTRACT_CLAIM = "CONTRACT_CLAIM"
    MEASURE_MISSING_OPERAND = "MEASURE_MISSING_OPERAND"
    NEW_PREPARATION_OR_CHART = "NEW_PREPARATION_OR_CHART"
    METHOD_POWER_STUDY = "METHOD_POWER_STUDY"
    REPAIR_ACTION_DELIVERY = "REPAIR_ACTION_DELIVERY"
    REDESIGN_REPRESENTATION_OR_DECISION = "REDESIGN_REPRESENTATION_OR_DECISION"
    NEW_POWERED_DESIGN = "NEW_POWERED_DESIGN"
    OBTAIN_AUTHORITY_OR_STOP = "OBTAIN_AUTHORITY_OR_STOP"
    PRESERVE_UNEVALUABLE = "PRESERVE_UNEVALUABLE"
    PREDECLARE_DISCRIMINATING_STUDY = "PREDECLARE_DISCRIMINATING_STUDY"
    NO_AUTOMATIC_NEXT_ACT = "NO_AUTOMATIC_NEXT_ACT"


class MetatheoryApplicability(StrEnum):
    REQUIRED = "REQUIRED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class MetatheoryMethodSelection(CanonicalRecord):
    """Exact static method/config binding; never executable dispatch."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-method-selection'

    selection_id: str
    capability_key: str
    capability_version: str
    config: ObjectIdentity
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.selection_id, field_name="selection_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")


__all__ = [
    'EvidenceDependenceClass',
    'LegitimateNextAct',
    'MetatheoryApplicability',
    'MetatheoryAggregateDisposition',
    'MetatheoryCellDisposition',
    'MetatheoryClaimKind',
    'MetatheoryEvidenceCeiling',
    'MetatheoryObstructionKind',
    'MetatheoryPredictiveLevel',
    'MetatheoryMethodSelection',
]
