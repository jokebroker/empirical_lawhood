'Outcome-blind predictive structural recurrence method for categorical structural recurrence.\n\nThe module owns only compact categorical records and pure deterministic\ntransformations.  It does not open target artifacts, inspect raw trajectories,\nissue authority, reveal outcomes, or execute a controller.  Target adapters\nremain responsible for lowering native evidence into the exact input and\nobservation records defined here.\n'

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from decimal import Decimal
from enum import StrEnum
from functools import reduce
from operator import mul
from typing import Any, ClassVar

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, EvidenceRung, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)


MAX_STRUCTURAL_RECURRENCE_RECORD_BYTES = 4 * 1024 * 1024


class UniversalRole(StrEnum):
    PREPARED_DENOMINATOR = "PREPARED_DENOMINATOR"
    RETAINED_HISTORY = "RETAINED_HISTORY"
    NATIVE_ACTION = "NATIVE_ACTION"
    ACTION_REALIZATION = "ACTION_REALIZATION"
    RECEIVER_TARGET = "RECEIVER_TARGET"
    RECEIVER_SINK = "RECEIVER_SINK"
    EFFORT = "EFFORT"
    VALIDITY = "VALIDITY"
    UNCERTAINTY = "UNCERTAINTY"
    PRESERVATION = "PRESERVATION"
    DYNAMICS = "DYNAMICS"
    REACHABILITY = "REACHABILITY"
    AUTHORITY = "AUTHORITY"
    SUPPORT = "SUPPORT"
    HOLD = "HOLD"
    HORIZON_CLOCK = "HORIZON_CLOCK"


UNIVERSAL_ROLES = tuple(sorted(UniversalRole, key=lambda value: value.value))

CORE_FIELD_IDS = (
    "admission_topology",
    "denominator_structure",
    "history_clock_quotient",
    'prospective_validation_disposition',
    "policy_branch",
    "support_transport",
)

RESTRICTION_FIELD_IDS = (
    "action_role",
    "boundary_role",
    "history_role",
    "internal_transfer_role",
    "numerical_role",
    "preparation_role",
    "receiver_role",
    "topology_mode_role",
)

ADMISSION_ROLES = (
    UniversalRole.SUPPORT,
    UniversalRole.RECEIVER_TARGET,
    UniversalRole.RECEIVER_SINK,
    UniversalRole.EFFORT,
    UniversalRole.VALIDITY,
    UniversalRole.UNCERTAINTY,
    UniversalRole.PRESERVATION,
    UniversalRole.DYNAMICS,
    UniversalRole.REACHABILITY,
    UniversalRole.AUTHORITY,
)

_REASON_PRIORITY = (
    UniversalRole.PREPARED_DENOMINATOR,
    UniversalRole.ACTION_REALIZATION,
    UniversalRole.RECEIVER_TARGET,
    UniversalRole.RECEIVER_SINK,
    UniversalRole.EFFORT,
    UniversalRole.VALIDITY,
    UniversalRole.UNCERTAINTY,
    UniversalRole.PRESERVATION,
    UniversalRole.DYNAMICS,
    UniversalRole.REACHABILITY,
    UniversalRole.AUTHORITY,
    UniversalRole.SUPPORT,
    UniversalRole.RETAINED_HISTORY,
    UniversalRole.HORIZON_CLOCK,
    UniversalRole.HOLD,
)


class TargetLevel(StrEnum):
    MEASUREMENT_READINESS = 'MEASUREMENT_READINESS'
    LAW_QUALIFICATION = 'LAW_QUALIFICATION'
    ADMISSION = 'ADMISSION'
    PROSPECTIVE_USE = 'PROSPECTIVE_USE'
    CONTROLLER_USE_CEILING = 'CONTROLLER_USE_CEILING'


class DenominatorStructure(StrEnum):
    COMMON_LAW = "COMMON_LAW"
    VIEW_LOCAL = "VIEW_LOCAL"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNEVALUABLE = "UNEVALUABLE"


class HistoryClockQuotient(StrEnum):
    CLOSE_CLOSED = "CLOSE_CLOSED"
    CLOSE_DIVERGED = "CLOSE_DIVERGED"
    TIMING_CONFOUNDED = "TIMING_CONFOUNDED"
    MIXED = "MIXED"
    UNEVALUABLE = "UNEVALUABLE"


class SupportTransport(StrEnum):
    EQUALITY = "EQUALITY"
    CONSERVATIVE_INCLUSION = "CONSERVATIVE_INCLUSION"
    BOUNDARY_LOSS = "BOUNDARY_LOSS"
    OPPOSED = "OPPOSED"
    UNEVALUABLE = "UNEVALUABLE"


class AdmissionTopology(StrEnum):
    EMPTY = "EMPTY"
    SINGLETON = "SINGLETON"
    MULTIPLE = "MULTIPLE"
    HOLD_ONLY = "HOLD_ONLY"
    UNEVALUABLE = "UNEVALUABLE"


class PolicyBranch(StrEnum):
    EXACT_ACTION = "EXACT_ACTION"
    HOLD = "HOLD"
    NONATTEMPT = "NONATTEMPT"
    UNEVALUABLE = "UNEVALUABLE"


class ProspectiveDisposition(StrEnum):
    VALIDATED = "VALIDATED"
    OPPOSED = "OPPOSED"
    CONDITION_FALSE = "CONDITION_FALSE"
    NONATTEMPT = "NONATTEMPT"
    UNEVALUABLE = "UNEVALUABLE"


class PreparationRole(StrEnum):
    DECISIVE = "DECISIVE"
    NONDECISIVE_ON_CHART = "NONDECISIVE_ON_CHART"
    UNEVALUABLE = "UNEVALUABLE"


class HistoryRole(StrEnum):
    RETAIN = "RETAIN"
    QUOTIENT_ON_DECLARED_PAIR = "QUOTIENT_ON_DECLARED_PAIR"
    TIMING_CONTROL_REQUIRED = "TIMING_CONTROL_REQUIRED"
    UNEVALUABLE = "UNEVALUABLE"


class ActionRole(StrEnum):
    REALIZATION_QUALIFIED = "REALIZATION_QUALIFIED"
    MISMATCH = "MISMATCH"
    ABSENT = "ABSENT"
    UNEVALUABLE = "UNEVALUABLE"


class ReceiverRole(StrEnum):
    TARGET_LIMITING = "TARGET_LIMITING"
    SINK_LIMITING = "SINK_LIMITING"
    PRESERVATION_LIMITING = "PRESERVATION_LIMITING"
    NONLIMITING = "NONLIMITING"
    UNEVALUABLE = "UNEVALUABLE"


class NumericalRole(StrEnum):
    QUALIFIED = "QUALIFIED"
    VIEW_LOCAL = "VIEW_LOCAL"
    INTERACTION = "INTERACTION"
    UNEVALUABLE = "UNEVALUABLE"


class BoundaryRole(StrEnum):
    INTERIOR = "INTERIOR"
    BOUNDARY_LIMITING = "BOUNDARY_LIMITING"
    SUPPORT_LOSS = "SUPPORT_LOSS"
    UNEVALUABLE = "UNEVALUABLE"


class InternalTransferRole(StrEnum):
    CLOSED = "CLOSED"
    RELEVANT_DIRECTION = "RELEVANT_DIRECTION"
    UNRESOLVED = "UNRESOLVED"
    UNEVALUABLE = "UNEVALUABLE"


class TopologyModeRole(StrEnum):
    DECISIVE = "DECISIVE"
    NONDECISIVE_ON_CHART = "NONDECISIVE_ON_CHART"
    INAPPLICABLE = "INAPPLICABLE"
    UNEVALUABLE = "UNEVALUABLE"


class OperandStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNEVALUABLE = "UNEVALUABLE"
    ABSENT = "ABSENT"


class ObserverStatus(StrEnum):
    QUALIFIED = "QUALIFIED"
    AMBIGUOUS = "AMBIGUOUS"
    ABSENT = "ABSENT"
    UNEVALUABLE = "UNEVALUABLE"


class DevelopmentControllerUseExpectation(StrEnum):
    VALIDATED = "VALIDATED"
    OPPOSED = "OPPOSED"
    CONDITION_FALSE = "CONDITION_FALSE"
    NONATTEMPT = "NONATTEMPT"
    UNEVALUABLE = "UNEVALUABLE"


class StructuralReason(StrEnum):
    NONE = "NONE"
    PREPARED_DENOMINATOR = "PREPARED_DENOMINATOR"
    RETAINED_HISTORY = "RETAINED_HISTORY"
    NATIVE_ACTION = "NATIVE_ACTION"
    ACTION_REALIZATION = "ACTION_REALIZATION"
    RECEIVER_TARGET = "RECEIVER_TARGET"
    RECEIVER_SINK = "RECEIVER_SINK"
    EFFORT = "EFFORT"
    VALIDITY = "VALIDITY"
    UNCERTAINTY = "UNCERTAINTY"
    PRESERVATION = "PRESERVATION"
    DYNAMICS = "DYNAMICS"
    REACHABILITY = "REACHABILITY"
    AUTHORITY = "AUTHORITY"
    SUPPORT = "SUPPORT"
    HOLD = "HOLD"
    HORIZON_CLOCK = "HORIZON_CLOCK"
    UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE = "UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE"
    PREDICTOR_NOT_RESTRICTIVE = "PREDICTOR_NOT_RESTRICTIVE"

    @classmethod
    def from_role(cls, role: UniversalRole) -> StructuralReason:
        return cls(role.value)


class PredictorKind(StrEnum):
    PRIMARY = "PRIMARY"
    WILDCARD = "WILDCARD"
    RESPONSE_ONLY = "RESPONSE_ONLY"
    HISTORY_FREE = "HISTORY_FREE"
    ALWAYS_ACT = "ALWAYS_ACT"
    ALWAYS_HOLD = "ALWAYS_HOLD"


class PredictionStatus(StrEnum):
    ISSUED = "ISSUED"
    PREDICTOR_NOT_RESTRICTIVE = "PREDICTOR_NOT_RESTRICTIVE"
    UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE = "UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE"


class MethodQualificationVerdict(StrEnum):
    METHOD_QUALIFIED = "METHOD_QUALIFIED"
    METHOD_FALSE_POSITIVE = "METHOD_FALSE_POSITIVE"
    METHOD_FALSE_NEGATIVE = "METHOD_FALSE_NEGATIVE"
    METHOD_NOT_RESTRICTIVE = "METHOD_NOT_RESTRICTIVE"
    METHOD_SCHEMA_INVALID = "METHOD_SCHEMA_INVALID"
    METHOD_UNEVALUABLE = "METHOD_UNEVALUABLE"


def _require_enum_set(
    values: tuple[StrEnum, ...],
    *,
    field_name: str,
    allow_empty: bool = False,
) -> None:
    if not values and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    expected = tuple(sorted(set(values), key=lambda value: value.value))
    if values != expected:
        raise ValueError(f"{field_name} must be sorted and unique")


def _product(values: tuple[int, ...]) -> int:
    return reduce(mul, values, 1)


def _prospective_space(level: TargetLevel) -> tuple[ProspectiveDisposition, ...]:
    if level in {TargetLevel.MEASUREMENT_READINESS, TargetLevel.LAW_QUALIFICATION, TargetLevel.ADMISSION}:
        return (ProspectiveDisposition.CONDITION_FALSE,)
    return tuple(sorted(ProspectiveDisposition, key=lambda value: value.value))


def _expected_observation_rung(level: TargetLevel) -> EvidenceRung:
    return {
        TargetLevel.MEASUREMENT_READINESS: EvidenceRung.MEASUREMENT,
        TargetLevel.LAW_QUALIFICATION: EvidenceRung.LOCAL_LAW,
        TargetLevel.ADMISSION: EvidenceRung.ADMISSION,
        TargetLevel.PROSPECTIVE_USE: EvidenceRung.CONTROLLER_USE,
        TargetLevel.CONTROLLER_USE_CEILING: EvidenceRung.CONTROLLER_USE,
    }[level]


@dataclass(frozen=True, slots=True)
class StructuralOntologyIdentity(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-ontology-identity'

    ontology_id: str
    ontology_version: str
    roles: tuple[UniversalRole, ...]
    core_field_ids: tuple[str, ...]
    restriction_field_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.ontology_id, field_name="ontology_id")
        validate_semantic_version(self.ontology_version)
        _require_enum_set(self.roles, field_name="roles")
        if self.roles != UNIVERSAL_ROLES:
            raise ValueError('categorical structural recurrence ontology must contain exactly the sixteen frozen roles')
        require_sorted_unique_strings(
            self.core_field_ids,
            field_name="core_field_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.restriction_field_ids,
            field_name="restriction_field_ids",
            allow_empty=False,
        )
        if self.core_field_ids != CORE_FIELD_IDS:
            raise ValueError('categorical structural recurrence core fields differ from the frozen six')
        if self.restriction_field_ids != RESTRICTION_FIELD_IDS:
            raise ValueError('categorical structural recurrence restriction fields differ from the frozen eight')


def structural_ontology() -> StructuralOntologyIdentity:
    return StructuralOntologyIdentity(
        ontology_id='categorical-structural-recurrence.ontology',
        ontology_version="1.0.0",
        roles=UNIVERSAL_ROLES,
        core_field_ids=CORE_FIELD_IDS,
        restriction_field_ids=RESTRICTION_FIELD_IDS,
    )


@dataclass(frozen=True, slots=True)
class DonorCorpusIdentity(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/donor-corpus-identity'

    corpus_id: str
    ontology: ObjectIdentity
    donor_handoffs: tuple[ObjectIdentity, ...]
    evidence_cutoff_rung: EvidenceRung
    evidence_cutoff_id: str
    decision_table_id: str
    decision_table_sha256: str
    excluded_target_slot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("corpus_id", self.corpus_id),
            ("evidence_cutoff_id", self.evidence_cutoff_id),
            ("decision_table_id", self.decision_table_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.donor_handoffs,
            attribute="object_id",
            field_name="donor_handoffs",
        )
        if not self.donor_handoffs:
            raise ValueError('structural recurrence donor corpus requires exact compact handoffs')
        if self.evidence_cutoff_rung not in {
            EvidenceRung.MEASUREMENT,
            EvidenceRung.ORDER_RELATION,
            EvidenceRung.RESPONSE,
            EvidenceRung.LOCAL_LAW,
        }:
            raise ValueError('structural recurrence predictor donor cutoff cannot include admission/prospective validation target outcomes')
        validate_sha256(self.decision_table_sha256, field_name="decision_table_sha256")
        require_sorted_unique_strings(
            self.excluded_target_slot_ids,
            field_name="excluded_target_slot_ids",
        )


@dataclass(frozen=True, slots=True)
class TargetSlotRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-slot-record'

    target_slot_id: str
    target_class_id: str
    canonical_order: int
    target_level: TargetLevel
    evidence_world_id: str
    source: ObjectIdentity
    ontology: ObjectIdentity
    reserve_source_ids: tuple[str, ...]
    outcome_naive_at_freeze: bool
    evaluation_outcomes_accessed: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("target_slot_id", self.target_slot_id),
            ("target_class_id", self.target_class_id),
            ("evidence_world_id", self.evidence_world_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.canonical_order < 1:
            raise ValueError("target canonical order must be positive")
        require_sorted_unique_strings(
            self.reserve_source_ids,
            field_name="reserve_source_ids",
        )
        if not self.outcome_naive_at_freeze or self.evaluation_outcomes_accessed:
            raise ValueError('structural recurrence target slots must freeze before protected outcomes')


@dataclass(frozen=True, slots=True)
class NativeRoleBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/native-role-binding'

    binding_id: str
    native_object_id: str
    role: UniversalRole
    native_unit: str
    native_direction: str
    clock_id: str
    missingness_rule: str
    claim_ceiling: EvidenceCeiling
    available: bool
    reuse_compatibility_id: str | None

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("native_object_id", self.native_object_id),
            ("clock_id", self.clock_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_nonempty(self.native_direction, field_name="native_direction")
        validate_nonempty(self.missingness_rule, field_name="missingness_rule")
        if self.reuse_compatibility_id is not None:
            validate_stable_id(
                self.reuse_compatibility_id,
                field_name="reuse_compatibility_id",
            )


@dataclass(frozen=True, slots=True)
class NativeRoleCompatibilityRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/native-role-compatibility-record'

    map_id: str
    target_slot_id: str
    ontology: ObjectIdentity
    bindings: tuple[NativeRoleBinding, ...]
    required_new_role_ids: tuple[str, ...]
    map_valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.map_id, field_name="map_id")
        validate_stable_id(self.target_slot_id, field_name="target_slot_id")
        require_sorted_unique_ids(
            self.bindings,
            attribute="binding_id",
            field_name="bindings",
        )
        if {binding.role for binding in self.bindings} != set(UniversalRole):
            raise ValueError("native compatibility map must account for all frozen roles")
        require_sorted_unique_strings(
            self.required_new_role_ids,
            field_name="required_new_role_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected_valid = not self.required_new_role_ids
        if self.map_valid != expected_valid:
            raise ValueError("new core roles make the native map an ontology counterexample")
        expected_reasons = () if expected_valid else ("UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE",)
        if self.reason_codes != expected_reasons:
            raise ValueError("native map reasons differ from its ontology disposition")
        by_native: dict[str, list[NativeRoleBinding]] = {}
        for binding in self.bindings:
            by_native.setdefault(binding.native_object_id, []).append(binding)
        for reused in by_native.values():
            roles = {binding.role for binding in reused}
            if len(roles) < 2:
                continue
            compatibility_ids = {binding.reuse_compatibility_id for binding in reused}
            if None in compatibility_ids or len(compatibility_ids) != 1:
                raise ValueError(
                    "one native coordinate serving several roles needs one explicit "
                    "noncircular compatibility identity"
                )

    @property
    def missing_roles(self) -> tuple[UniversalRole, ...]:
        return tuple(
            role
            for role in UNIVERSAL_ROLES
            if not any(binding.role is role and binding.available for binding in self.bindings)
        )


@dataclass(frozen=True, slots=True)
class AdmissionOperandFact(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/admission-operand-fact'

    operand_id: str
    role: UniversalRole
    status: OperandStatus
    evidence: ObjectIdentity
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.operand_id, field_name="operand_id")
        if self.role not in ADMISSION_ROLES:
            raise ValueError("admission operand uses a role outside the frozen intersection")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is OperandStatus.PASS:
            if self.reason_codes:
                raise ValueError("passing admission operand cannot retain failures")
        elif not self.reason_codes:
            raise ValueError("nonpassing admission operand requires a reason")


@dataclass(frozen=True, slots=True)
class ActionCandidateFact(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-candidate-fact'

    action_id: str
    development_rank: int
    target_response_positive: bool | None
    action_realization: ActionRole
    operands: tuple[AdmissionOperandFact, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.action_id, field_name="action_id")
        if self.development_rank < 1:
            raise ValueError("action development rank must be positive")
        require_sorted_unique_ids(
            self.operands,
            attribute="operand_id",
            field_name="operands",
        )
        observed_roles = tuple(operand.role for operand in self.operands)
        if observed_roles != ADMISSION_ROLES:
            raise ValueError(
                "each action requires the exact ordered ten-role noncompensating intersection"
            )

    @property
    def admitted(self) -> bool:
        return self.action_realization is ActionRole.REALIZATION_QUALIFIED and all(
            operand.status is OperandStatus.PASS for operand in self.operands
        )

    @property
    def unresolved(self) -> bool:
        return self.action_realization is ActionRole.UNEVALUABLE or any(
            operand.status in {OperandStatus.UNEVALUABLE, OperandStatus.ABSENT}
            for operand in self.operands
        )

    def operand(self, role: UniversalRole) -> AdmissionOperandFact:
        return next(value for value in self.operands if value.role is role)


@dataclass(frozen=True, slots=True)
class StructuralRestrictionProfile(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-restriction-profile'

    preparation_role: PreparationRole
    history_role: HistoryRole
    action_role: ActionRole
    receiver_role: ReceiverRole
    numerical_role: NumericalRole
    boundary_role: BoundaryRole
    internal_transfer_role: InternalTransferRole
    topology_mode_role: TopologyModeRole


@dataclass(frozen=True, slots=True)
class StructuralPredictionInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-prediction-input'

    input_id: str
    ontology: StructuralOntologyIdentity
    donor_corpus: DonorCorpusIdentity
    target_slot: TargetSlotRecord
    compatibility: NativeRoleCompatibilityRecord
    denominator_evidence: DenominatorStructure
    history_evidence: HistoryClockQuotient
    support_evidence: SupportTransport
    restriction_evidence: StructuralRestrictionProfile
    action_candidates: tuple[ActionCandidateFact, ...]
    hold_only_chart: bool
    deterministic_selector_available: bool
    observer_status: ObserverStatus
    source_ready: bool
    receiver_available: bool
    prospective_units_available: bool
    hold_preservation_expected: bool
    checkpoint_receiver_close: bool | None
    prospective_validation_expectation: DevelopmentControllerUseExpectation
    missing_measurement_roles: tuple[UniversalRole, ...]
    required_new_role_ids: tuple[str, ...]
    secondary_reason_codes: tuple[StructuralReason, ...]
    independent_development_unit_count: int
    evidence_rung: EvidenceRung
    outcome_access: OutcomeAccess
    raw_trajectory_values_present: bool
    native_threshold_values_present: bool
    target_admission_outcomes_accessed: bool
    target_prospective_validation_outcomes_accessed: bool
    post_reveal_diagnostics_accessed: bool
    source_readiness_handoff: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        ontology_identity = ObjectIdentity.from_record(
            self.ontology.ontology_id,
            self.ontology,
        )
        if (
            self.donor_corpus.ontology != ontology_identity
            or self.target_slot.ontology != ontology_identity
            or self.compatibility.ontology != ontology_identity
        ):
            raise ValueError('structural recurrence operands use different ontology bytes')
        if (
            self.compatibility.target_slot_id != self.target_slot.target_slot_id
            or self.target_slot.target_slot_id not in self.donor_corpus.excluded_target_slot_ids
        ):
            raise ValueError("target slot is not excluded from its donor corpus")
        require_sorted_unique_ids(
            self.action_candidates,
            attribute="action_id",
            field_name="action_candidates",
        )
        if self.hold_only_chart:
            if self.action_candidates:
                raise ValueError("a hold-only chart cannot contain non-hold action candidates")
        elif not self.action_candidates:
            raise ValueError("a non-hold chart requires finite action candidates")
        _require_enum_set(self.missing_measurement_roles, field_name='missing_measurement_roles', allow_empty=True)
        if self.missing_measurement_roles != self.compatibility.missing_roles:
            raise ValueError("declared measurement missing roles differ from the native compatibility map")
        require_sorted_unique_strings(
            self.required_new_role_ids,
            field_name="required_new_role_ids",
        )
        if self.required_new_role_ids != self.compatibility.required_new_role_ids:
            raise ValueError("new-role requirements differ from the compatibility map")
        _require_enum_set(
            self.secondary_reason_codes,
            field_name="secondary_reason_codes",
            allow_empty=True,
        )
        if len(self.secondary_reason_codes) > 2:
            raise ValueError("a structural prediction may retain at most two secondary reasons")
        if self.independent_development_unit_count < 0:
            raise ValueError("independent development unit count cannot be negative")
        source_readiness_stop = self.independent_development_unit_count == 0
        if source_readiness_stop:
            if (
                self.target_slot.target_level is not TargetLevel.MEASUREMENT_READINESS
                or self.evidence_rung is not EvidenceRung.MEASUREMENT
                or self.source_ready
                or self.source_readiness_handoff is None
            ):
                raise ValueError(
                    'zero development units require an exact measurement readiness source-readiness-stop handoff'
                )
        elif self.source_readiness_handoff is not None:
            raise ValueError('source-readiness-stop handoff is valid only for a zero-unit measurement readiness stop')
        if self.evidence_rung not in {
            EvidenceRung.MEASUREMENT,
            EvidenceRung.ORDER_RELATION,
            EvidenceRung.RESPONSE,
            EvidenceRung.LOCAL_LAW,
        }:
            raise ValueError('protected target admission/prospective validation results cannot enter prediction input')
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        }:
            raise ValueError('structural recurrence prediction input cannot contain protected target outcomes')
        if any(
            (
                self.raw_trajectory_values_present,
                self.native_threshold_values_present,
                self.target_admission_outcomes_accessed,
                self.target_prospective_validation_outcomes_accessed,
                self.post_reveal_diagnostics_accessed,
            )
        ):
            raise ValueError('structural recurrence prediction input contains forbidden outcome or numeric leakage')
        missing = set(self.missing_measurement_roles)
        unevaluable_witnesses = (
            (
                self.denominator_evidence is DenominatorStructure.UNEVALUABLE,
                {
                    UniversalRole.PREPARED_DENOMINATOR,
                    UniversalRole.VALIDITY,
                },
            ),
            (
                self.history_evidence is HistoryClockQuotient.UNEVALUABLE,
                {
                    UniversalRole.RETAINED_HISTORY,
                    UniversalRole.HORIZON_CLOCK,
                },
            ),
            (
                self.support_evidence is SupportTransport.UNEVALUABLE,
                {UniversalRole.SUPPORT},
            ),
            (
                self.prospective_validation_expectation is DevelopmentControllerUseExpectation.UNEVALUABLE,
                {
                    UniversalRole.ACTION_REALIZATION,
                    UniversalRole.RECEIVER_TARGET,
                    UniversalRole.RECEIVER_SINK,
                    UniversalRole.PRESERVATION,
                    UniversalRole.VALIDITY,
                    UniversalRole.AUTHORITY,
                },
            ),
        )
        if any(
            required and not (missing & witnesses) for required, witnesses in unevaluable_witnesses
        ):
            raise ValueError(
                "UNEVALUABLE prediction evidence requires its specific missing measurement role"
            )
        if any(
            role in ADMISSION_ROLES and candidate.operand(role).status is OperandStatus.PASS
            for role in missing
            for candidate in self.action_candidates
        ):
            raise ValueError("missing measurement admission operand cannot be marked passing")


@dataclass(frozen=True, slots=True)
class StructuralCorePrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-core-prediction'

    denominator_structure: tuple[DenominatorStructure, ...]
    history_clock_quotient: tuple[HistoryClockQuotient, ...]
    support_transport: tuple[SupportTransport, ...]
    admission_topology: tuple[AdmissionTopology, ...]
    policy_branch: tuple[PolicyBranch, ...]
    prospective_validation_disposition: tuple[ProspectiveDisposition, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "denominator_structure",
            "history_clock_quotient",
            "support_transport",
            "admission_topology",
            "policy_branch",
            'prospective_validation_disposition',
        ):
            _require_enum_set(getattr(self, field_name), field_name=field_name)

    @property
    def widths(self) -> tuple[int, ...]:
        return (
            len(self.denominator_structure),
            len(self.history_clock_quotient),
            len(self.support_transport),
            len(self.admission_topology),
            len(self.policy_branch),
            len(self.prospective_validation_disposition),
        )

    @property
    def singleton_count(self) -> int:
        return sum(width == 1 for width in self.widths)

    @property
    def contains_unevaluable(self) -> bool:
        return any(
            (
                DenominatorStructure.UNEVALUABLE in self.denominator_structure,
                HistoryClockQuotient.UNEVALUABLE in self.history_clock_quotient,
                SupportTransport.UNEVALUABLE in self.support_transport,
                AdmissionTopology.UNEVALUABLE in self.admission_topology,
                PolicyBranch.UNEVALUABLE in self.policy_branch,
                ProspectiveDisposition.UNEVALUABLE in self.prospective_validation_disposition,
            )
        )


@dataclass(frozen=True, slots=True)
class StructuralSafetyPrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-safety-prediction'

    policy_branch: tuple[PolicyBranch, ...]
    primary_reason: tuple[StructuralReason, ...]
    action_realization: tuple[ActionRole, ...]
    hold_preservation: tuple[bool, ...]
    prospective_validation_condition_false: tuple[bool, ...]
    admission_direction_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_enum_set(self.policy_branch, field_name="policy_branch")
        _require_enum_set(self.primary_reason, field_name="primary_reason")
        _require_enum_set(self.action_realization, field_name="action_realization")
        if (
            not self.hold_preservation
            or tuple(sorted(set(self.hold_preservation))) != self.hold_preservation
        ):
            raise ValueError("hold_preservation must be sorted and unique")
        if (
            not self.prospective_validation_condition_false
            or tuple(sorted(set(self.prospective_validation_condition_false))) != self.prospective_validation_condition_false
        ):
            raise ValueError('prospective_validation_condition_false must be sorted and unique')
        require_sorted_unique_strings(
            self.admission_direction_ids,
            field_name="admission_direction_ids",
            allow_empty=False,
        )

    @property
    def singleton(self) -> bool:
        return all(
            len(values) == 1
            for values in (
                self.policy_branch,
                self.primary_reason,
                self.action_realization,
                self.hold_preservation,
                self.prospective_validation_condition_false,
                self.admission_direction_ids,
            )
        )


@dataclass(frozen=True, slots=True)
class StructuralPrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-prediction'

    prediction_id: str
    target_slot_id: str
    target_level: TargetLevel
    evidence_world_id: str
    prediction_input: ObjectIdentity
    ontology: ObjectIdentity
    donor_corpus: ObjectIdentity
    predictor_kind: PredictorKind
    status: PredictionStatus
    core: StructuralCorePrediction
    restrictions: StructuralRestrictionProfile
    safety: StructuralSafetyPrediction
    missing_measurement_roles: tuple[UniversalRole, ...]
    required_new_role_ids: tuple[str, ...]
    secondary_reason_codes: tuple[StructuralReason, ...]
    valid_state_count: int
    predicted_state_count: int
    excluded_state_count: int
    sharpness: Decimal
    singleton_core_count: int
    core_width_passed: bool
    safety_singleton_passed: bool
    reason_set_passed: bool
    sharpness_passed: bool
    issued_before_target_outcomes: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("prediction_id", self.prediction_id),
            ("target_slot_id", self.target_slot_id),
            ("evidence_world_id", self.evidence_world_id),
        ):
            validate_stable_id(value, field_name=name)
        _require_enum_set(self.missing_measurement_roles, field_name='missing_measurement_roles', allow_empty=True)
        require_sorted_unique_strings(
            self.required_new_role_ids,
            field_name="required_new_role_ids",
        )
        _require_enum_set(
            self.secondary_reason_codes,
            field_name="secondary_reason_codes",
            allow_empty=True,
        )
        if len(self.secondary_reason_codes) > 2:
            raise ValueError("prediction retains too many secondary reasons")
        valid_spaces = (
            tuple(DenominatorStructure),
            tuple(HistoryClockQuotient),
            tuple(SupportTransport),
            tuple(AdmissionTopology),
            tuple(PolicyBranch),
            _prospective_space(self.target_level),
        )
        if not set(self.core.prospective_validation_disposition) <= set(valid_spaces[-1]):
            raise ValueError('prospective validation prediction exceeds target-level condition-false pruning')
        expected_valid = _product(tuple(len(values) for values in valid_spaces))
        expected_predicted = _product(self.core.widths)
        expected_excluded = expected_valid - expected_predicted
        expected_sharpness = Decimal(expected_excluded) / Decimal(expected_valid)
        if (
            self.valid_state_count != expected_valid
            or self.predicted_state_count != expected_predicted
            or self.excluded_state_count != expected_excluded
            or self.sharpness != expected_sharpness
        ):
            raise ValueError("prediction sharpness is not derived from its exact state space")
        validate_decimal(
            self.sharpness,
            field_name="sharpness",
            minimum=Decimal(0),
        )
        opposition_separated = not {
            ProspectiveDisposition.VALIDATED,
            ProspectiveDisposition.OPPOSED,
        }.issubset(self.core.prospective_validation_disposition)
        expected_width = (
            self.core.singleton_count >= 4 and max(self.core.widths) <= 2 and opposition_separated
        )
        expected_safety = self.safety.singleton
        expected_reasons = (
            len(self.safety.primary_reason) == 1 and len(self.secondary_reason_codes) <= 2
        )
        expected_sharpness_pass = self.sharpness >= Decimal("0.50")
        if (
            self.singleton_core_count != self.core.singleton_count
            or self.core_width_passed != expected_width
            or self.safety_singleton_passed != expected_safety
            or self.reason_set_passed != expected_reasons
            or self.sharpness_passed != expected_sharpness_pass
        ):
            raise ValueError("prediction restrictiveness flags are not mechanically derived")
        if (
            self.core.contains_unevaluable
            and not self.missing_measurement_roles
            and self.predictor_kind is not PredictorKind.WILDCARD
        ):
            raise ValueError("UNEVALUABLE cannot be used as an ungrounded wildcard")
        if self.predictor_kind is PredictorKind.PRIMARY:
            reason_required = (
                PolicyBranch.HOLD in self.core.policy_branch
                or PolicyBranch.NONATTEMPT in self.core.policy_branch
                or PolicyBranch.UNEVALUABLE in self.core.policy_branch
                or ProspectiveDisposition.OPPOSED in self.core.prospective_validation_disposition
                or ProspectiveDisposition.NONATTEMPT in self.core.prospective_validation_disposition
                or ProspectiveDisposition.UNEVALUABLE in self.core.prospective_validation_disposition
                or self.core.contains_unevaluable
            )
            if reason_required and self.safety.primary_reason == (StructuralReason.NONE,):
                raise ValueError(
                    "hold/nonattempt/opposed/unevaluable primary prediction needs a reason"
                )
        restrictive = (
            expected_width and expected_safety and expected_reasons and expected_sharpness_pass
        )
        if self.status is PredictionStatus.UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE:
            if self.predictor_kind is not PredictorKind.PRIMARY or not self.required_new_role_ids:
                raise ValueError("ontology counterexample lacks an exact new-role witness")
        elif self.required_new_role_ids:
            raise ValueError("new-role witness cannot be silently accepted")
        elif restrictive != (self.status is PredictionStatus.ISSUED):
            raise ValueError("prediction status differs from the frozen sharpness contract")
        if not self.issued_before_target_outcomes:
            raise ValueError("post-outcome predictions are ineligible")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        }:
            raise ValueError("issued prediction has protected outcome access")


@dataclass(frozen=True, slots=True)
class StructuralObservedCore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-observed-core'

    denominator_structure: DenominatorStructure
    history_clock_quotient: HistoryClockQuotient
    support_transport: SupportTransport
    admission_topology: AdmissionTopology
    policy_branch: PolicyBranch
    prospective_validation_disposition: ProspectiveDisposition


@dataclass(frozen=True, slots=True)
class StructuralSafetyObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-safety-observation'

    policy_branch: PolicyBranch
    primary_reason: StructuralReason
    action_realization: ActionRole
    hold_preservation: bool
    prospective_validation_condition_false: bool
    admission_direction_id: str

    def __post_init__(self) -> None:
        validate_stable_id(
            self.admission_direction_id,
            field_name="admission_direction_id",
        )


@dataclass(frozen=True, slots=True)
class StructuralObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-observation'

    observation_id: str
    target_slot_id: str
    target_level: TargetLevel
    evidence_world_id: str
    target_handoff: ObjectIdentity
    evaluator: ObjectIdentity
    core: StructuralObservedCore
    restrictions: StructuralRestrictionProfile
    safety: StructuralSafetyObservation
    ontology_counterexample: bool
    required_new_role_ids: tuple[str, ...]
    independent_unit_count: int
    observed_rung: EvidenceRung
    outcome_access: OutcomeAccess
    source_readiness_handoff: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("observation_id", self.observation_id),
            ("target_slot_id", self.target_slot_id),
            ("evidence_world_id", self.evidence_world_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.required_new_role_ids,
            field_name="required_new_role_ids",
        )
        if self.ontology_counterexample != bool(self.required_new_role_ids):
            raise ValueError("observed ontology status differs from its new-role witness")
        if self.independent_unit_count < 0:
            raise ValueError("independent target unit count cannot be negative")
        source_readiness_stop = self.independent_unit_count == 0
        if source_readiness_stop:
            if (
                self.target_level is not TargetLevel.MEASUREMENT_READINESS
                or self.observed_rung is not EvidenceRung.MEASUREMENT
                or self.source_readiness_handoff is None
                or self.source_readiness_handoff != self.target_handoff
            ):
                raise ValueError(
                    'zero target units require the exact measurement readiness source-readiness-stop handoff'
                )
        elif self.source_readiness_handoff is not None:
            raise ValueError('source-readiness-stop handoff is valid only for a zero-unit measurement readiness stop')
        if self.observed_rung is not _expected_observation_rung(self.target_level):
            raise ValueError("structural observation rung differs from target eligibility level")
        if self.outcome_access not in {
            OutcomeAccess.EVALUATOR_REVEAL,
            OutcomeAccess.PRIVILEGED_TRUTH,
        }:
            raise ValueError("structural observation requires evaluator-scoped truth access")
        if (
            self.target_level in {TargetLevel.MEASUREMENT_READINESS, TargetLevel.LAW_QUALIFICATION, TargetLevel.ADMISSION}
            and self.core.prospective_validation_disposition is not ProspectiveDisposition.CONDITION_FALSE
        ):
            raise ValueError('prospective validation must be condition-false below prospective use')


@dataclass(frozen=True, slots=True)
class PredictiveFieldMatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/predictive-field-match'

    field_id: str
    predicted_values: tuple[str, ...]
    observed_value: str
    matched: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.field_id, field_name="field_id")
        require_sorted_unique_strings(
            self.predicted_values,
            field_name="predicted_values",
            allow_empty=False,
        )
        validate_nonempty(self.observed_value, field_name="observed_value")
        if self.matched != (self.observed_value in self.predicted_values):
            raise ValueError("field match is not derived from prediction membership")


@dataclass(frozen=True, slots=True)
class PredictiveMatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/predictive-match'

    match_id: str
    target_slot_id: str
    predictor_kind: PredictorKind
    prediction: ObjectIdentity
    observation: ObjectIdentity
    field_matches: tuple[PredictiveFieldMatch, ...]
    coverage: bool
    sharpness: Decimal
    safety_exact: bool
    reason_exact: bool
    ontology_counterexample_exact: bool
    false_action_count: int
    hidden_sink_admission_count: int
    missed_opportunity_count: int
    false_history_quotient_count: int
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.match_id, field_name="match_id")
        validate_stable_id(self.target_slot_id, field_name="target_slot_id")
        require_sorted_unique_ids(
            self.field_matches,
            attribute="field_id",
            field_name="field_matches",
        )
        if tuple(value.field_id for value in self.field_matches) != CORE_FIELD_IDS:
            raise ValueError("predictive match must retain all six core fields")
        if self.coverage != all(value.matched for value in self.field_matches):
            raise ValueError("coverage is not the conjunction of core field matches")
        validate_decimal(self.sharpness, field_name="sharpness", minimum=Decimal(0))
        for name, value in (
            ("false_action_count", self.false_action_count),
            ("hidden_sink_admission_count", self.hidden_sink_admission_count),
            ("missed_opportunity_count", self.missed_opportunity_count),
            ("false_history_quotient_count", self.false_history_quotient_count),
        ):
            if value not in {0, 1}:
                raise ValueError(f"{name} must be a per-target binary count")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class PredictiveComparatorPanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/predictive-comparator-panel'

    panel_id: str
    prediction_input: ObjectIdentity
    predictions: tuple[StructuralPrediction, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        require_sorted_unique_ids(
            self.predictions,
            attribute="prediction_id",
            field_name="predictions",
        )
        expected = {
            PredictorKind.WILDCARD,
            PredictorKind.RESPONSE_ONLY,
            PredictorKind.HISTORY_FREE,
            PredictorKind.ALWAYS_ACT,
            PredictorKind.ALWAYS_HOLD,
        }
        if {value.predictor_kind for value in self.predictions} != expected:
            raise ValueError("comparator panel differs from the frozen five comparators")
        if any(value.prediction_input != self.prediction_input for value in self.predictions):
            raise ValueError("comparator panel crosses prediction inputs")


@dataclass(frozen=True, slots=True)
class PredictiveRecurrenceAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/predictive-recurrence-adjudication'

    adjudication_id: str
    target_slot_ids: tuple[str, ...]
    primary_match_ids: tuple[str, ...]
    comparator_match_ids: tuple[str, ...]
    primary_conjunction_passed: bool
    comparator_superiority_passed: bool
    ontology_counterexample_count: int
    false_action_count: int
    hidden_sink_admission_count: int
    missed_opportunity_count: int
    verdict: MethodQualificationVerdict
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        for field_name, values in (
            ("target_slot_ids", self.target_slot_ids),
            ("primary_match_ids", self.primary_match_ids),
            ("comparator_match_ids", self.comparator_match_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        for name, value in (
            ("ontology_counterexample_count", self.ontology_counterexample_count),
            ("false_action_count", self.false_action_count),
            ("hidden_sink_admission_count", self.hidden_sink_admission_count),
            ("missed_opportunity_count", self.missed_opportunity_count),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


def _primary_reason_for_empty(input_record: StructuralPredictionInput) -> StructuralReason:
    for role in _REASON_PRIORITY:
        if role in input_record.missing_measurement_roles:
            return StructuralReason.from_role(role)
        if role in ADMISSION_ROLES and any(
            candidate.operand(role).status is not OperandStatus.PASS
            for candidate in input_record.action_candidates
        ):
            return StructuralReason.from_role(role)
    return StructuralReason.HOLD


def _ranked_candidates(
    input_record: StructuralPredictionInput,
    *,
    admitted_only: bool,
    positive_only: bool = False,
) -> tuple[ActionCandidateFact, ...]:
    values = tuple(
        candidate
        for candidate in input_record.action_candidates
        if (not admitted_only or candidate.admitted)
        and (not positive_only or candidate.target_response_positive is True)
    )
    return tuple(sorted(values, key=lambda value: (-value.development_rank, value.action_id)))


def _admission_topology(input_record: StructuralPredictionInput) -> AdmissionTopology:
    if input_record.hold_only_chart:
        return AdmissionTopology.HOLD_ONLY
    admitted = _ranked_candidates(input_record, admitted_only=True)
    if len(admitted) == 1:
        return AdmissionTopology.SINGLETON
    if len(admitted) > 1:
        return AdmissionTopology.MULTIPLE
    if any(candidate.unresolved for candidate in input_record.action_candidates):
        return AdmissionTopology.UNEVALUABLE
    return AdmissionTopology.EMPTY


def _critical_nonattempt_reason(
    input_record: StructuralPredictionInput,
) -> StructuralReason | None:
    if not input_record.source_ready:
        return StructuralReason.VALIDITY
    if input_record.denominator_evidence in {
        DenominatorStructure.INCOMPATIBLE,
        DenominatorStructure.UNEVALUABLE,
    }:
        return StructuralReason.PREPARED_DENOMINATOR
    if not input_record.receiver_available:
        return StructuralReason.RECEIVER_TARGET
    realization_states = {
        candidate.action_realization for candidate in input_record.action_candidates
    }
    if realization_states and ActionRole.REALIZATION_QUALIFIED not in realization_states:
        return StructuralReason.ACTION_REALIZATION
    if input_record.action_candidates and all(
        candidate.operand(UniversalRole.AUTHORITY).status is not OperandStatus.PASS
        for candidate in input_record.action_candidates
    ):
        return StructuralReason.AUTHORITY
    if (
        input_record.target_slot.target_level in {TargetLevel.PROSPECTIVE_USE, TargetLevel.CONTROLLER_USE_CEILING}
        and not input_record.prospective_units_available
    ):
        return StructuralReason.VALIDITY
    return None


def _derive_primary_policy(
    input_record: StructuralPredictionInput,
    topology: AdmissionTopology,
) -> tuple[PolicyBranch, ActionCandidateFact | None, StructuralReason]:
    nonattempt_reason = _critical_nonattempt_reason(input_record)
    if nonattempt_reason is not None:
        return PolicyBranch.NONATTEMPT, None, nonattempt_reason
    admitted = _ranked_candidates(input_record, admitted_only=True)
    if topology in {AdmissionTopology.SINGLETON, AdmissionTopology.MULTIPLE}:
        if input_record.observer_status is not ObserverStatus.QUALIFIED:
            return PolicyBranch.HOLD, None, StructuralReason.VALIDITY
        if not input_record.deterministic_selector_available:
            return PolicyBranch.HOLD, None, StructuralReason.NATIVE_ACTION
        return PolicyBranch.EXACT_ACTION, admitted[0], StructuralReason.NONE
    if topology is AdmissionTopology.HOLD_ONLY:
        return PolicyBranch.HOLD, None, StructuralReason.HOLD
    return PolicyBranch.HOLD, None, _primary_reason_for_empty(input_record)


def _derive_prospective(
    input_record: StructuralPredictionInput,
    policy: PolicyBranch,
) -> ProspectiveDisposition:
    level = input_record.target_slot.target_level
    if level in {TargetLevel.MEASUREMENT_READINESS, TargetLevel.LAW_QUALIFICATION, TargetLevel.ADMISSION}:
        return ProspectiveDisposition.CONDITION_FALSE
    if policy is PolicyBranch.NONATTEMPT:
        return ProspectiveDisposition.NONATTEMPT
    if policy is not PolicyBranch.EXACT_ACTION:
        return ProspectiveDisposition.CONDITION_FALSE
    if not input_record.hold_preservation_expected:
        return ProspectiveDisposition.OPPOSED
    return {
        DevelopmentControllerUseExpectation.VALIDATED: ProspectiveDisposition.VALIDATED,
        DevelopmentControllerUseExpectation.OPPOSED: ProspectiveDisposition.OPPOSED,
        DevelopmentControllerUseExpectation.CONDITION_FALSE: ProspectiveDisposition.CONDITION_FALSE,
        DevelopmentControllerUseExpectation.NONATTEMPT: ProspectiveDisposition.NONATTEMPT,
        DevelopmentControllerUseExpectation.UNEVALUABLE: ProspectiveDisposition.UNEVALUABLE,
    }[input_record.prospective_validation_expectation]


def _prospective_or_unevaluable_reason(
    input_record: StructuralPredictionInput,
    prospective_validation: ProspectiveDisposition,
) -> StructuralReason:
    if not input_record.hold_preservation_expected:
        return StructuralReason.PRESERVATION
    receiver_role = input_record.restriction_evidence.receiver_role
    if receiver_role is ReceiverRole.SINK_LIMITING:
        return StructuralReason.RECEIVER_SINK
    if receiver_role is ReceiverRole.PRESERVATION_LIMITING:
        return StructuralReason.PRESERVATION
    if receiver_role is ReceiverRole.TARGET_LIMITING:
        return StructuralReason.RECEIVER_TARGET
    if prospective_validation is ProspectiveDisposition.OPPOSED:
        return StructuralReason.RECEIVER_TARGET
    if input_record.missing_measurement_roles:
        return StructuralReason.from_role(input_record.missing_measurement_roles[0])
    return StructuralReason.VALIDITY


def _core(
    *,
    denominator: tuple[DenominatorStructure, ...],
    history: tuple[HistoryClockQuotient, ...],
    support: tuple[SupportTransport, ...],
    admission: tuple[AdmissionTopology, ...],
    policy: tuple[PolicyBranch, ...],
    prospective_validation: tuple[ProspectiveDisposition, ...],
) -> StructuralCorePrediction:
    return StructuralCorePrediction(
        denominator_structure=tuple(sorted(denominator, key=lambda value: value.value)),
        history_clock_quotient=tuple(sorted(history, key=lambda value: value.value)),
        support_transport=tuple(sorted(support, key=lambda value: value.value)),
        admission_topology=tuple(sorted(admission, key=lambda value: value.value)),
        policy_branch=tuple(sorted(policy, key=lambda value: value.value)),
        prospective_validation_disposition=tuple(sorted(prospective_validation, key=lambda value: value.value)),
    )


def _safety(
    *,
    policy: tuple[PolicyBranch, ...],
    reasons: tuple[StructuralReason, ...],
    realization: tuple[ActionRole, ...],
    hold_preservation: tuple[bool, ...],
    prospective_validation_condition_false: tuple[bool, ...],
    direction_ids: tuple[str, ...],
) -> StructuralSafetyPrediction:
    return StructuralSafetyPrediction(
        policy_branch=tuple(sorted(policy, key=lambda value: value.value)),
        primary_reason=tuple(sorted(reasons, key=lambda value: value.value)),
        action_realization=tuple(sorted(realization, key=lambda value: value.value)),
        hold_preservation=tuple(sorted(hold_preservation)),
        prospective_validation_condition_false=tuple(sorted(prospective_validation_condition_false)),
        admission_direction_ids=tuple(sorted(direction_ids)),
    )


def _make_prediction(
    input_record: StructuralPredictionInput,
    *,
    predictor_kind: PredictorKind,
    core: StructuralCorePrediction,
    restrictions: StructuralRestrictionProfile,
    safety: StructuralSafetyPrediction,
    required_new_role_ids: tuple[str, ...] = (),
) -> StructuralPrediction:
    valid_state_count = _product(
        (
            len(DenominatorStructure),
            len(HistoryClockQuotient),
            len(SupportTransport),
            len(AdmissionTopology),
            len(PolicyBranch),
            len(_prospective_space(input_record.target_slot.target_level)),
        )
    )
    predicted_state_count = _product(core.widths)
    excluded_state_count = valid_state_count - predicted_state_count
    sharpness = Decimal(excluded_state_count) / Decimal(valid_state_count)
    core_width_passed = (
        core.singleton_count >= 4
        and max(core.widths) <= 2
        and not {
            ProspectiveDisposition.VALIDATED,
            ProspectiveDisposition.OPPOSED,
        }.issubset(core.prospective_validation_disposition)
    )
    safety_singleton_passed = safety.singleton
    reason_set_passed = (
        len(safety.primary_reason) == 1 and len(input_record.secondary_reason_codes) <= 2
    )
    sharpness_passed = sharpness >= Decimal("0.50")
    restrictive = (
        core_width_passed and safety_singleton_passed and reason_set_passed and sharpness_passed
    )
    if required_new_role_ids:
        status = PredictionStatus.UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE
    elif restrictive:
        status = PredictionStatus.ISSUED
    else:
        status = PredictionStatus.PREDICTOR_NOT_RESTRICTIVE
    return StructuralPrediction(
        prediction_id=(
            f"structural-recurrence.prediction.{predictor_kind.value.lower().replace('_', '-')}"
            f".{input_record.fingerprint()[:24]}"
        ),
        target_slot_id=input_record.target_slot.target_slot_id,
        target_level=input_record.target_slot.target_level,
        evidence_world_id=input_record.target_slot.evidence_world_id,
        prediction_input=ObjectIdentity.from_record(
            input_record.input_id,
            input_record,
        ),
        ontology=ObjectIdentity.from_record(
            input_record.ontology.ontology_id,
            input_record.ontology,
        ),
        donor_corpus=ObjectIdentity.from_record(
            input_record.donor_corpus.corpus_id,
            input_record.donor_corpus,
        ),
        predictor_kind=predictor_kind,
        status=status,
        core=core,
        restrictions=restrictions,
        safety=safety,
        missing_measurement_roles=input_record.missing_measurement_roles,
        required_new_role_ids=required_new_role_ids,
        secondary_reason_codes=input_record.secondary_reason_codes,
        valid_state_count=valid_state_count,
        predicted_state_count=predicted_state_count,
        excluded_state_count=excluded_state_count,
        sharpness=sharpness,
        singleton_core_count=core.singleton_count,
        core_width_passed=core_width_passed,
        safety_singleton_passed=safety_singleton_passed,
        reason_set_passed=reason_set_passed,
        sharpness_passed=sharpness_passed,
        issued_before_target_outcomes=True,
        outcome_access=input_record.outcome_access,
    )


def _primary_components(
    input_record: StructuralPredictionInput,
) -> tuple[
    StructuralCorePrediction,
    StructuralRestrictionProfile,
    StructuralSafetyPrediction,
]:
    topology = _admission_topology(input_record)
    policy, selected, reason = _derive_primary_policy(input_record, topology)
    prospective_validation = _derive_prospective(input_record, policy)
    if reason is StructuralReason.NONE and prospective_validation in {
        ProspectiveDisposition.OPPOSED,
        ProspectiveDisposition.NONATTEMPT,
        ProspectiveDisposition.UNEVALUABLE,
    }:
        reason = _prospective_or_unevaluable_reason(input_record, prospective_validation)
    if reason is StructuralReason.NONE and (
        input_record.denominator_evidence is DenominatorStructure.UNEVALUABLE
        or input_record.history_evidence is HistoryClockQuotient.UNEVALUABLE
        or input_record.support_evidence is SupportTransport.UNEVALUABLE
    ):
        reason = _prospective_or_unevaluable_reason(input_record, ProspectiveDisposition.UNEVALUABLE)
    action_role = (
        selected.action_realization
        if selected is not None
        else input_record.restriction_evidence.action_role
    )
    direction = (
        selected.action_id
        if selected is not None
        else ("nonattempt" if policy is PolicyBranch.NONATTEMPT else "hold")
    )
    core = _core(
        denominator=(input_record.denominator_evidence,),
        history=(input_record.history_evidence,),
        support=(input_record.support_evidence,),
        admission=(topology,),
        policy=(policy,),
        prospective_validation=(prospective_validation,),
    )
    safety = _safety(
        policy=(policy,),
        reasons=(reason,),
        realization=(action_role,),
        hold_preservation=(input_record.hold_preservation_expected,),
        prospective_validation_condition_false=(prospective_validation is ProspectiveDisposition.CONDITION_FALSE,),
        direction_ids=(direction,),
    )
    return core, input_record.restriction_evidence, safety


class PredictiveStructuralRecurrenceCompiler:
    "Compile one sharp structural prediction from categorical facts spanning measurement through law qualification."

    def compile(self, input_record: StructuralPredictionInput) -> StructuralPrediction:
        core, restrictions, safety = _primary_components(input_record)
        if input_record.required_new_role_ids:
            policy = PolicyBranch.NONATTEMPT
            prospective_validation = (
                ProspectiveDisposition.CONDITION_FALSE
                if input_record.target_slot.target_level
                in {TargetLevel.MEASUREMENT_READINESS, TargetLevel.LAW_QUALIFICATION, TargetLevel.ADMISSION}
                else ProspectiveDisposition.NONATTEMPT
            )
            core = _core(
                denominator=core.denominator_structure,
                history=core.history_clock_quotient,
                support=core.support_transport,
                admission=(AdmissionTopology.HOLD_ONLY,),
                policy=(policy,),
                prospective_validation=(prospective_validation,),
            )
            safety = _safety(
                policy=(policy,),
                reasons=(StructuralReason.UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE,),
                realization=(restrictions.action_role,),
                hold_preservation=(True,),
                prospective_validation_condition_false=(prospective_validation is ProspectiveDisposition.CONDITION_FALSE,),
                direction_ids=("nonattempt",),
            )
        return _make_prediction(
            input_record,
            predictor_kind=PredictorKind.PRIMARY,
            core=core,
            restrictions=restrictions,
            safety=safety,
            required_new_role_ids=input_record.required_new_role_ids,
        )

    def comparator_panel(
        self,
        input_record: StructuralPredictionInput,
    ) -> PredictiveComparatorPanel:
        predictions = tuple(
            sorted(
                (
                    wildcard_comparator(input_record),
                    response_only_comparator(input_record),
                    history_free_comparator(input_record),
                    always_act_comparator(input_record),
                    always_hold_comparator(input_record),
                ),
                key=lambda value: value.prediction_id,
            )
        )
        return PredictiveComparatorPanel(
            panel_id=f"structural-recurrence.comparators.{input_record.fingerprint()[:24]}",
            prediction_input=ObjectIdentity.from_record(
                input_record.input_id,
                input_record,
            ),
            predictions=predictions,
        )


def wildcard_comparator(input_record: StructuralPredictionInput) -> StructuralPrediction:
    core = _core(
        denominator=tuple(DenominatorStructure),
        history=tuple(HistoryClockQuotient),
        support=tuple(SupportTransport),
        admission=tuple(AdmissionTopology),
        policy=tuple(PolicyBranch),
        prospective_validation=_prospective_space(input_record.target_slot.target_level),
    )
    safety = _safety(
        policy=tuple(PolicyBranch),
        reasons=tuple(StructuralReason),
        realization=tuple(ActionRole),
        hold_preservation=(False, True),
        prospective_validation_condition_false=(False, True),
        direction_ids=tuple(
            sorted(
                {
                    "hold",
                    "nonattempt",
                    *(candidate.action_id for candidate in input_record.action_candidates),
                }
            )
        ),
    )
    return _make_prediction(
        input_record,
        predictor_kind=PredictorKind.WILDCARD,
        core=core,
        restrictions=input_record.restriction_evidence,
        safety=safety,
    )


def response_only_comparator(
    input_record: StructuralPredictionInput,
) -> StructuralPrediction:
    primary_core, restrictions, _ = _primary_components(input_record)
    candidates = _ranked_candidates(
        input_record,
        admitted_only=False,
        positive_only=True,
    )
    if candidates:
        selected = candidates[0]
        policy = PolicyBranch.EXACT_ACTION
        admission = AdmissionTopology.SINGLETON
        prospective_validation = (
            ProspectiveDisposition.VALIDATED
            if input_record.target_slot.target_level in {TargetLevel.PROSPECTIVE_USE, TargetLevel.CONTROLLER_USE_CEILING}
            else ProspectiveDisposition.CONDITION_FALSE
        )
        reason = StructuralReason.NONE
        direction = selected.action_id
        realization = selected.action_realization
    else:
        policy = PolicyBranch.HOLD
        admission = AdmissionTopology.EMPTY
        prospective_validation = ProspectiveDisposition.CONDITION_FALSE
        reason = StructuralReason.RECEIVER_TARGET
        direction = "hold"
        realization = restrictions.action_role
    core = _core(
        denominator=primary_core.denominator_structure,
        history=primary_core.history_clock_quotient,
        support=primary_core.support_transport,
        admission=(admission,),
        policy=(policy,),
        prospective_validation=(prospective_validation,),
    )
    safety = _safety(
        policy=(policy,),
        reasons=(reason,),
        realization=(realization,),
        hold_preservation=(input_record.hold_preservation_expected,),
        prospective_validation_condition_false=(prospective_validation is ProspectiveDisposition.CONDITION_FALSE,),
        direction_ids=(direction,),
    )
    return _make_prediction(
        input_record,
        predictor_kind=PredictorKind.RESPONSE_ONLY,
        core=core,
        restrictions=restrictions,
        safety=safety,
    )


def history_free_comparator(
    input_record: StructuralPredictionInput,
) -> StructuralPrediction:
    primary_core, restrictions, safety = _primary_components(input_record)
    history = (
        HistoryClockQuotient.CLOSE_CLOSED
        if input_record.checkpoint_receiver_close is True
        else primary_core.history_clock_quotient[0]
    )
    history_role = (
        HistoryRole.QUOTIENT_ON_DECLARED_PAIR
        if input_record.checkpoint_receiver_close is True
        else restrictions.history_role
    )
    changed_restrictions = StructuralRestrictionProfile(
        preparation_role=restrictions.preparation_role,
        history_role=history_role,
        action_role=restrictions.action_role,
        receiver_role=restrictions.receiver_role,
        numerical_role=restrictions.numerical_role,
        boundary_role=restrictions.boundary_role,
        internal_transfer_role=restrictions.internal_transfer_role,
        topology_mode_role=restrictions.topology_mode_role,
    )
    core = _core(
        denominator=primary_core.denominator_structure,
        history=(history,),
        support=primary_core.support_transport,
        admission=primary_core.admission_topology,
        policy=primary_core.policy_branch,
        prospective_validation=primary_core.prospective_validation_disposition,
    )
    return _make_prediction(
        input_record,
        predictor_kind=PredictorKind.HISTORY_FREE,
        core=core,
        restrictions=changed_restrictions,
        safety=safety,
    )


def always_act_comparator(input_record: StructuralPredictionInput) -> StructuralPrediction:
    primary_core, restrictions, _ = _primary_components(input_record)
    candidates = _ranked_candidates(input_record, admitted_only=False)
    if not candidates:
        return always_hold_comparator(input_record, kind=PredictorKind.ALWAYS_ACT)
    selected = candidates[0]
    prospective_validation = (
        ProspectiveDisposition.VALIDATED
        if input_record.target_slot.target_level in {TargetLevel.PROSPECTIVE_USE, TargetLevel.CONTROLLER_USE_CEILING}
        else ProspectiveDisposition.CONDITION_FALSE
    )
    policy = PolicyBranch.EXACT_ACTION
    core = _core(
        denominator=primary_core.denominator_structure,
        history=primary_core.history_clock_quotient,
        support=primary_core.support_transport,
        admission=(AdmissionTopology.SINGLETON,),
        policy=(policy,),
        prospective_validation=(prospective_validation,),
    )
    safety = _safety(
        policy=(policy,),
        reasons=(StructuralReason.NONE,),
        realization=(selected.action_realization,),
        hold_preservation=(input_record.hold_preservation_expected,),
        prospective_validation_condition_false=(prospective_validation is ProspectiveDisposition.CONDITION_FALSE,),
        direction_ids=(selected.action_id,),
    )
    return _make_prediction(
        input_record,
        predictor_kind=PredictorKind.ALWAYS_ACT,
        core=core,
        restrictions=restrictions,
        safety=safety,
    )


def always_hold_comparator(
    input_record: StructuralPredictionInput,
    *,
    kind: PredictorKind = PredictorKind.ALWAYS_HOLD,
) -> StructuralPrediction:
    primary_core, restrictions, _ = _primary_components(input_record)
    policy = PolicyBranch.HOLD
    prospective_validation = ProspectiveDisposition.CONDITION_FALSE
    core = _core(
        denominator=primary_core.denominator_structure,
        history=primary_core.history_clock_quotient,
        support=primary_core.support_transport,
        admission=(AdmissionTopology.HOLD_ONLY,),
        policy=(policy,),
        prospective_validation=(prospective_validation,),
    )
    safety = _safety(
        policy=(policy,),
        reasons=(StructuralReason.HOLD,),
        realization=(restrictions.action_role,),
        hold_preservation=(True,),
        prospective_validation_condition_false=(True,),
        direction_ids=("hold",),
    )
    return _make_prediction(
        input_record,
        predictor_kind=kind,
        core=core,
        restrictions=restrictions,
        safety=safety,
    )


class PredictiveStructuralMatcher:
    """Match one issued prediction to one independently derived signature."""

    def match(
        self,
        prediction: StructuralPrediction,
        observation: StructuralObservation,
    ) -> PredictiveMatch:
        if (
            prediction.target_slot_id != observation.target_slot_id
            or prediction.target_level is not observation.target_level
            or prediction.evidence_world_id != observation.evidence_world_id
        ):
            raise ValueError("prediction and observation target scope differ")
        predicted = {
            "denominator_structure": tuple(
                value.value for value in prediction.core.denominator_structure
            ),
            "history_clock_quotient": tuple(
                value.value for value in prediction.core.history_clock_quotient
            ),
            "support_transport": tuple(value.value for value in prediction.core.support_transport),
            "admission_topology": tuple(
                value.value for value in prediction.core.admission_topology
            ),
            "policy_branch": tuple(value.value for value in prediction.core.policy_branch),
            'prospective_validation_disposition': tuple(value.value for value in prediction.core.prospective_validation_disposition),
        }
        observed = {
            "denominator_structure": observation.core.denominator_structure.value,
            "history_clock_quotient": observation.core.history_clock_quotient.value,
            "support_transport": observation.core.support_transport.value,
            "admission_topology": observation.core.admission_topology.value,
            "policy_branch": observation.core.policy_branch.value,
            'prospective_validation_disposition': observation.core.prospective_validation_disposition.value,
        }
        field_matches = tuple(
            PredictiveFieldMatch(
                field_id=field_id,
                predicted_values=tuple(sorted(predicted[field_id])),
                observed_value=observed[field_id],
                matched=observed[field_id] in predicted[field_id],
            )
            for field_id in CORE_FIELD_IDS
        )
        safety = prediction.safety
        safety_exact = (
            safety.policy_branch == (observation.safety.policy_branch,)
            and safety.primary_reason == (observation.safety.primary_reason,)
            and safety.action_realization == (observation.safety.action_realization,)
            and safety.hold_preservation == (observation.safety.hold_preservation,)
            and safety.prospective_validation_condition_false == (observation.safety.prospective_validation_condition_false,)
            and safety.admission_direction_ids == (observation.safety.admission_direction_id,)
        )
        reason_exact = safety.primary_reason == (observation.safety.primary_reason,)
        ontology_exact = (
            prediction.status is PredictionStatus.UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE
            and observation.ontology_counterexample
            and prediction.required_new_role_ids == observation.required_new_role_ids
        )
        predicts_exact_action = PolicyBranch.EXACT_ACTION in prediction.core.policy_branch
        direction_exact = safety.admission_direction_ids == (
            observation.safety.admission_direction_id,
        )
        observed_action_safe = (
            observation.core.policy_branch is PolicyBranch.EXACT_ACTION
            and observation.core.admission_topology
            in {AdmissionTopology.SINGLETON, AdmissionTopology.MULTIPLE}
            and observation.restrictions.action_role is ActionRole.REALIZATION_QUALIFIED
            and observation.restrictions.receiver_role
            not in {
                ReceiverRole.SINK_LIMITING,
                ReceiverRole.PRESERVATION_LIMITING,
            }
            and observation.core.support_transport
            not in {SupportTransport.OPPOSED, SupportTransport.UNEVALUABLE}
        )
        false_action = int(
            predicts_exact_action and (not observed_action_safe or not direction_exact)
        )
        hidden_sink = int(
            predicts_exact_action
            and observation.restrictions.receiver_role
            in {ReceiverRole.SINK_LIMITING, ReceiverRole.PRESERVATION_LIMITING}
        )
        observed_opportunity = observation.core.policy_branch is PolicyBranch.EXACT_ACTION
        predicted_opportunity_exact = (
            prediction.core.policy_branch == (PolicyBranch.EXACT_ACTION,) and direction_exact
        )
        missed_opportunity = int(observed_opportunity and not predicted_opportunity_exact)
        false_history = int(
            prediction.core.history_clock_quotient == (HistoryClockQuotient.CLOSE_CLOSED,)
            and observation.core.history_clock_quotient is not HistoryClockQuotient.CLOSE_CLOSED
        )
        coverage = all(value.matched for value in field_matches)
        ordinary_pass = (
            coverage
            and prediction.status is PredictionStatus.ISSUED
            and prediction.sharpness >= Decimal("0.50")
            and safety_exact
            and reason_exact
        )
        passed = ontology_exact if observation.ontology_counterexample else ordinary_pass
        reasons: set[str] = set()
        if not coverage:
            reasons.add("CORE_COVERAGE_FAILED")
        if not prediction.sharpness_passed:
            reasons.add("PREDICTION_NOT_RESTRICTIVE")
        if not safety_exact:
            reasons.add("SAFETY_SINGLETON_MISMATCH")
        if not reason_exact:
            reasons.add("PRIMARY_REASON_MISMATCH")
        if observation.ontology_counterexample and not ontology_exact:
            reasons.add("ONTOLOGY_COUNTEREXAMPLE_MISSED")
        if false_action:
            reasons.add("FALSE_ACTION")
        if hidden_sink:
            reasons.add("HIDDEN_SINK_ADMISSION")
        if missed_opportunity:
            reasons.add("TARGET_OPPORTUNITY_MISSED")
        if false_history:
            reasons.add("FALSE_HISTORY_QUOTIENT")
        return PredictiveMatch(
            match_id=(
                f"structural-recurrence.match.{prediction.predictor_kind.value.lower().replace('_', '-')}"
                f".{observation.observation_id}"
            ),
            target_slot_id=observation.target_slot_id,
            predictor_kind=prediction.predictor_kind,
            prediction=ObjectIdentity.from_record(
                prediction.prediction_id,
                prediction,
            ),
            observation=ObjectIdentity.from_record(
                observation.observation_id,
                observation,
            ),
            field_matches=field_matches,
            coverage=coverage,
            sharpness=prediction.sharpness,
            safety_exact=safety_exact,
            reason_exact=reason_exact,
            ontology_counterexample_exact=ontology_exact,
            false_action_count=false_action,
            hidden_sink_admission_count=hidden_sink,
            missed_opportunity_count=missed_opportunity,
            false_history_quotient_count=false_history,
            passed=passed,
            reason_codes=tuple(sorted(reasons)),
        )


class PredictiveStructuralAdjudicator:
    """Adjudicate truth-known method conformance without pooling targets."""

    def adjudicate(
        self,
        *,
        adjudication_id: str,
        matches: tuple[PredictiveMatch, ...],
    ) -> PredictiveRecurrenceAdjudication:
        validate_stable_id(adjudication_id, field_name="adjudication_id")
        require_sorted_unique_ids(matches, attribute="match_id", field_name="matches")
        target_slot_ids = tuple(sorted({value.target_slot_id for value in matches}))
        expected_kinds = set(PredictorKind)
        by_target: dict[str, dict[PredictorKind, PredictiveMatch]] = {}
        for match in matches:
            target = by_target.setdefault(match.target_slot_id, {})
            if match.predictor_kind in target:
                raise ValueError("target has duplicate predictor matches")
            target[match.predictor_kind] = match
        if not by_target or any(set(values) != expected_kinds for values in by_target.values()):
            raise ValueError(
                "adjudication requires primary plus five comparator matches per target"
            )
        primary = tuple(values[PredictorKind.PRIMARY] for values in by_target.values())
        comparators = tuple(
            value
            for values in by_target.values()
            for kind, value in values.items()
            if kind is not PredictorKind.PRIMARY
        )
        primary_pass = all(value.passed for value in primary)
        primary_false_actions = sum(value.false_action_count for value in primary)
        primary_hidden_sinks = sum(value.hidden_sink_admission_count for value in primary)
        primary_missed = sum(value.missed_opportunity_count for value in primary)
        primary_false_history = sum(value.false_history_quotient_count for value in primary)

        def values(kind: PredictorKind) -> tuple[PredictiveMatch, ...]:
            return tuple(target[kind] for target in by_target.values())

        wildcard = values(PredictorKind.WILDCARD)
        response_only = values(PredictorKind.RESPONSE_ONLY)
        history_free = values(PredictorKind.HISTORY_FREE)
        always_act = values(PredictorKind.ALWAYS_ACT)
        always_hold = values(PredictorKind.ALWAYS_HOLD)
        sharper_than_wildcard = all(
            left.sharpness > right.sharpness for left, right in zip(primary, wildcard, strict=True)
        )
        no_extra_false_acts = primary_false_actions <= sum(
            value.false_action_count for value in always_hold
        )
        hold_missed = sum(value.missed_opportunity_count for value in always_hold)
        opportunity_superiority = hold_missed == 0 or primary_missed < hold_missed
        response_sink_errors = sum(value.hidden_sink_admission_count for value in response_only)
        always_act_sink_errors = sum(value.hidden_sink_admission_count for value in always_act)
        sink_superiority = (
            response_sink_errors == 0 or primary_hidden_sinks < response_sink_errors
        ) and (always_act_sink_errors == 0 or primary_hidden_sinks < always_act_sink_errors)
        history_errors = sum(value.false_history_quotient_count for value in history_free)
        history_superiority = history_errors == 0 or primary_false_history < history_errors
        comparator_superiority = all(
            (
                sharper_than_wildcard,
                no_extra_false_acts,
                opportunity_superiority,
                sink_superiority,
                history_superiority,
            )
        )
        if any(
            not value.sharpness >= Decimal("0.50")
            or "PREDICTION_NOT_RESTRICTIVE" in value.reason_codes
            for value in primary
        ):
            verdict = MethodQualificationVerdict.METHOD_NOT_RESTRICTIVE
        elif primary_false_actions or primary_hidden_sinks:
            verdict = MethodQualificationVerdict.METHOD_FALSE_POSITIVE
        elif primary_missed:
            verdict = MethodQualificationVerdict.METHOD_FALSE_NEGATIVE
        elif primary_pass and comparator_superiority:
            verdict = MethodQualificationVerdict.METHOD_QUALIFIED
        else:
            verdict = MethodQualificationVerdict.METHOD_UNEVALUABLE
        reason_codes: set[str] = set()
        if not primary_pass:
            reason_codes.add("PRIMARY_CONJUNCTION_FAILED")
        if not comparator_superiority:
            reason_codes.add("COMPARATOR_SUPERIORITY_FAILED")
        if verdict is MethodQualificationVerdict.METHOD_NOT_RESTRICTIVE:
            reason_codes.add("PREDICTOR_NOT_RESTRICTIVE")
        if verdict is MethodQualificationVerdict.METHOD_FALSE_POSITIVE:
            reason_codes.add("METHOD_FALSE_POSITIVE")
        if verdict is MethodQualificationVerdict.METHOD_FALSE_NEGATIVE:
            reason_codes.add("METHOD_FALSE_NEGATIVE")
        return PredictiveRecurrenceAdjudication(
            adjudication_id=adjudication_id,
            target_slot_ids=target_slot_ids,
            primary_match_ids=tuple(sorted(value.match_id for value in primary)),
            comparator_match_ids=tuple(sorted(value.match_id for value in comparators)),
            primary_conjunction_passed=primary_pass,
            comparator_superiority_passed=comparator_superiority,
            ontology_counterexample_count=sum(
                value.ontology_counterexample_exact for value in primary
            ),
            false_action_count=primary_false_actions,
            hidden_sink_admission_count=primary_hidden_sinks,
            missed_opportunity_count=primary_missed,
            verdict=verdict,
            reason_codes=tuple(sorted(reason_codes)),
        )


class DonorFamily(StrEnum):
    TRUTH_KNOWN_CONFORMANCE = "TRUTH_KNOWN_CONFORMANCE"
    GYM_TORAX_ADMISSION_CONTROLLER_USE = 'GYM_TORAX_ADMISSION_CONTROLLER_USE'
    PYBAMM_ADMISSION_CONTROLLER_USE_RESPONSE_CLOSURE = 'PYBAMM_ADMISSION_CONTROLLER_USE_RESPONSE_CLOSURE'
    BOPTEST_REALIZED_ACTION_STOP = "BOPTEST_REALIZED_ACTION_STOP"
    QUANTUM_TRAJECTORY_QUALIFICATION_AND_RESPONSE = 'QUANTUM_TRAJECTORY_QUALIFICATION_AND_RESPONSE'
    HELDOUT_STRUCTURAL_LOCAL_LAW = 'HELDOUT_STRUCTURAL_LOCAL_LAW'


REQUIRED_DONOR_FAMILIES = (
    DonorFamily.BOPTEST_REALIZED_ACTION_STOP,
    DonorFamily.GYM_TORAX_ADMISSION_CONTROLLER_USE,
    DonorFamily.PYBAMM_ADMISSION_CONTROLLER_USE_RESPONSE_CLOSURE,
    DonorFamily.HELDOUT_STRUCTURAL_LOCAL_LAW,
    DonorFamily.QUANTUM_TRAJECTORY_QUALIFICATION_AND_RESPONSE,
    DonorFamily.TRUTH_KNOWN_CONFORMANCE,
)


class StructuralRecurrenceTruthKnownFixtureKind(StrEnum):
    POSITIVE_LOCAL_SINGLETON = "POSITIVE_LOCAL_SINGLETON"
    MULTIPLE_ACTION_DETERMINISTIC_SELECTION = "MULTIPLE_ACTION_DETERMINISTIC_SELECTION"
    POSITIVE_RESPONSE_HIDDEN_SINK = "POSITIVE_RESPONSE_HIDDEN_SINK"
    EMPTY_ADMISSION = 'EMPTY_ADMISSION'
    MISSING_REALIZED_ACTION = "MISSING_REALIZED_ACTION"
    MANDATORY_HOLD = "MANDATORY_HOLD"
    PROSPECTIVE_VALIDATION_SUCCESS = 'PROSPECTIVE_VALIDATION_SUCCESS'
    PROSPECTIVE_VALIDATION_OPPOSITION = 'PROSPECTIVE_VALIDATION_OPPOSITION'
    UNSUPPORTED_DENOMINATOR = "UNSUPPORTED_DENOMINATOR"
    LOSSY_COMPATIBILITY_MAP = "LOSSY_COMPATIBILITY_MAP"
    NEW_ONTOLOGY_ROLE = "NEW_ONTOLOGY_ROLE"


REQUIRED_TRUTH_KNOWN_FIXTURE_KINDS = (
    StructuralRecurrenceTruthKnownFixtureKind.EMPTY_ADMISSION,
    StructuralRecurrenceTruthKnownFixtureKind.LOSSY_COMPATIBILITY_MAP,
    StructuralRecurrenceTruthKnownFixtureKind.MANDATORY_HOLD,
    StructuralRecurrenceTruthKnownFixtureKind.MISSING_REALIZED_ACTION,
    StructuralRecurrenceTruthKnownFixtureKind.MULTIPLE_ACTION_DETERMINISTIC_SELECTION,
    StructuralRecurrenceTruthKnownFixtureKind.NEW_ONTOLOGY_ROLE,
    StructuralRecurrenceTruthKnownFixtureKind.PROSPECTIVE_VALIDATION_OPPOSITION,
    StructuralRecurrenceTruthKnownFixtureKind.PROSPECTIVE_VALIDATION_SUCCESS,
    StructuralRecurrenceTruthKnownFixtureKind.POSITIVE_LOCAL_SINGLETON,
    StructuralRecurrenceTruthKnownFixtureKind.POSITIVE_RESPONSE_HIDDEN_SINK,
    StructuralRecurrenceTruthKnownFixtureKind.UNSUPPORTED_DENOMINATOR,
)


class HeldoutTargetKind(StrEnum):
    HELDOUT_QUANTUM = "HELDOUT_QUANTUM"
    NESTED_QUANTUM_NEGATIVE_CONTROL = "NESTED_QUANTUM_NEGATIVE_CONTROL"
    HELDOUT_CROSS_SUBSTRATE = "HELDOUT_CROSS_SUBSTRATE"


class NativeControllerUseTerminal(StrEnum):
    VALIDATED = "VALIDATED"
    OPPOSED = "OPPOSED"
    CONDITION_FALSE = "CONDITION_FALSE"
    NONATTEMPT = "NONATTEMPT"
    UNEVALUABLE = "UNEVALUABLE"
    SOURCE_STOP = "SOURCE_STOP"


class HeldoutStructuralVerdict(StrEnum):
    HELDOUT_ADMISSION_CONTROL_RECURRENCE_SUPPORTED = (
        "HELDOUT_ADMISSION_CONTROL_RECURRENCE_SUPPORTED"
    )
    HELDOUT_PROCEDURAL_RECURRENCE_ONLY = "HELDOUT_PROCEDURAL_RECURRENCE_ONLY"
    HELDOUT_SAFE_ABSTENTION_CALIBRATED = "HELDOUT_SAFE_ABSTENTION_CALIBRATED"
    STRUCTURAL_COMPILER_COUNTEREXAMPLE = "STRUCTURAL_COMPILER_COUNTEREXAMPLE"
    HELDOUT_TARGETS_UNEVALUABLE = "HELDOUT_TARGETS_UNEVALUABLE"


_RUNG_INDEX = {rung: index for index, rung in enumerate(EvidenceRung)}
_TARGET_LEVEL_INDEX = {level: index for index, level in enumerate(TargetLevel)}


@dataclass(frozen=True, slots=True)
class DonorEvidenceRecord(CanonicalRecord):
    """One compact donor-visible terminal record, never a held-out member."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/donor-evidence-record'

    donor_id: str
    family: DonorFamily
    substrate_id: str
    evidence_world_id: str
    controlling_handoff: ObjectIdentity
    terminal_rung: EvidenceRung
    terminal_code: str
    response_handoff: ObjectIdentity | None
    law_qualification_handoff: ObjectIdentity | None
    admission_handoff: ObjectIdentity | None
    prospective_validation_handoff: ObjectIdentity | None
    outcome_access: OutcomeAccess
    heldout_member_eligible: bool
    heldout_independent_unit_count: int
    scientific_payload_values_present: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("donor_id", self.donor_id),
            ("substrate_id", self.substrate_id),
            ("evidence_world_id", self.evidence_world_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.terminal_code, field_name="terminal_code")
        if self.outcome_access not in {
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.PRIVILEGED_TRUTH,
        }:
            raise ValueError("donor outcomes must be explicitly visible")
        if self.heldout_member_eligible or self.heldout_independent_unit_count != 0:
            raise ValueError("donor evidence contributes zero held-out members and units")
        if self.scientific_payload_values_present:
            raise ValueError("donor manifest may contain identities, not scientific payload values")
        handoffs = {
            EvidenceRung.RESPONSE: self.response_handoff,
            EvidenceRung.LOCAL_LAW: self.law_qualification_handoff,
            EvidenceRung.ADMISSION: self.admission_handoff,
            EvidenceRung.CONTROLLER_USE: self.prospective_validation_handoff,
        }
        for rung, handoff in handoffs.items():
            required = _RUNG_INDEX[self.terminal_rung] >= _RUNG_INDEX[rung]
            if required != (handoff is not None):
                raise ValueError(
                    "donor rung chain must contain exactly every attained component qualification--prospective validation handoff"
                )
        prospective_validation_families = {
            DonorFamily.TRUTH_KNOWN_CONFORMANCE,
            DonorFamily.GYM_TORAX_ADMISSION_CONTROLLER_USE,
            DonorFamily.PYBAMM_ADMISSION_CONTROLLER_USE_RESPONSE_CLOSURE,
        }
        if self.family in prospective_validation_families and self.terminal_rung is not EvidenceRung.CONTROLLER_USE:
            raise ValueError("control donor family requires a complete native component qualification--prospective validation handoff")


@dataclass(frozen=True, slots=True)
class StructuralDonorCorpus(CanonicalRecord):
    """Exact visible donor corpus with a separate target-input cutoff."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-donor-corpus'

    corpus_id: str
    ontology: ObjectIdentity
    donor_records: tuple[DonorEvidenceRecord, ...]
    donor_visible_maximum_rung: EvidenceRung
    target_prediction_input_cutoff_rung: EvidenceRung
    excluded_target_slot_ids: tuple[str, ...]
    heldout_member_count: int
    heldout_independent_unit_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.corpus_id, field_name="corpus_id")
        require_sorted_unique_ids(
            self.donor_records,
            attribute="donor_id",
            field_name="donor_records",
        )
        if {record.family for record in self.donor_records} != set(REQUIRED_DONOR_FAMILIES):
            raise ValueError('donor corpus must contain every frozen structural donor family')
        maximum = max(
            (record.terminal_rung for record in self.donor_records),
            key=_RUNG_INDEX.__getitem__,
        )
        if self.donor_visible_maximum_rung is not maximum:
            raise ValueError("donor-visible maximum rung is not derived from donor records")
        if self.target_prediction_input_cutoff_rung is not EvidenceRung.LOCAL_LAW:
            raise ValueError("held-out target compiler inputs remain capped at development law qualification")
        require_sorted_unique_strings(
            self.excluded_target_slot_ids,
            field_name="excluded_target_slot_ids",
            allow_empty=False,
        )
        if self.heldout_member_count != 0 or self.heldout_independent_unit_count != 0:
            raise ValueError("donor corpus contributes zero held-out members and units")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("the frozen donor corpus is development-visible")

    @property
    def controlling_handoffs(self) -> tuple[ObjectIdentity, ...]:
        return tuple(
            sorted(
                (record.controlling_handoff for record in self.donor_records),
                key=lambda value: value.object_id,
            )
        )


def make_donor_evidence_record(
    *,
    donor_id: str,
    family: DonorFamily,
    substrate_id: str,
    evidence_world_id: str,
    controlling_handoff: ObjectIdentity,
    terminal_rung: EvidenceRung,
    terminal_code: str,
    response_handoff: ObjectIdentity | None,
    law_qualification_handoff: ObjectIdentity | None,
    admission_handoff: ObjectIdentity | None,
    prospective_validation_handoff: ObjectIdentity | None,
    outcome_access: OutcomeAccess,
) -> DonorEvidenceRecord:
    """Construct one identity-only donor terminal without payload or membership."""

    return DonorEvidenceRecord(
        donor_id=donor_id,
        family=family,
        substrate_id=substrate_id,
        evidence_world_id=evidence_world_id,
        controlling_handoff=controlling_handoff,
        terminal_rung=terminal_rung,
        terminal_code=terminal_code,
        response_handoff=response_handoff,
        law_qualification_handoff=law_qualification_handoff,
        admission_handoff=admission_handoff,
        prospective_validation_handoff=prospective_validation_handoff,
        outcome_access=outcome_access,
        heldout_member_eligible=False,
        heldout_independent_unit_count=0,
        scientific_payload_values_present=False,
    )


def build_structural_recurrence_donor_corpus(
    *,
    corpus_id: str,
    ontology: StructuralOntologyIdentity,
    excluded_target_slot_ids: tuple[str, ...],
    truth_known_conformance: DonorEvidenceRecord,
    gym_torax: DonorEvidenceRecord,
    pybamm: DonorEvidenceRecord,
    boptest: DonorEvidenceRecord,
    quantum_trajectory_qualification_and_response: DonorEvidenceRecord,
    heldout_structural_local_law: DonorEvidenceRecord,
) -> StructuralDonorCorpus:
    """Bind the six predeclared donor families with no held-out contribution."""

    supplied = {
        DonorFamily.TRUTH_KNOWN_CONFORMANCE: truth_known_conformance,
        DonorFamily.GYM_TORAX_ADMISSION_CONTROLLER_USE: gym_torax,
        DonorFamily.PYBAMM_ADMISSION_CONTROLLER_USE_RESPONSE_CLOSURE: pybamm,
        DonorFamily.BOPTEST_REALIZED_ACTION_STOP: boptest,
        DonorFamily.QUANTUM_TRAJECTORY_QUALIFICATION_AND_RESPONSE: quantum_trajectory_qualification_and_response,
        DonorFamily.HELDOUT_STRUCTURAL_LOCAL_LAW: heldout_structural_local_law,
    }
    if any(record.family is not family for family, record in supplied.items()):
        raise ValueError("donor argument is assigned to the wrong frozen family")
    records = tuple(sorted(supplied.values(), key=lambda value: value.donor_id))
    return StructuralDonorCorpus(
        corpus_id=corpus_id,
        ontology=ObjectIdentity.from_record(ontology.ontology_id, ontology),
        donor_records=records,
        donor_visible_maximum_rung=max(
            (record.terminal_rung for record in records),
            key=_RUNG_INDEX.__getitem__,
        ),
        target_prediction_input_cutoff_rung=EvidenceRung.LOCAL_LAW,
        excluded_target_slot_ids=tuple(sorted(excluded_target_slot_ids)),
        heldout_member_count=0,
        heldout_independent_unit_count=0,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


@dataclass(frozen=True, slots=True)
class TargetLocalityAudit(CanonicalRecord):
    """Fail-closed ownership and collision audit for one target-local design."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-locality-audit'

    audit_id: str
    target_slot_id: str
    target_kind: HeldoutTargetKind
    parent_target_slot_id: str | None
    threshold_owner_target_slot_id: str
    coefficient_owner_target_slot_id: str
    sample_owner_target_slot_id: str
    likelihood_owner_target_slot_id: str
    action_owner_target_slot_id: str
    shared_unit_target_slot_ids: tuple[str, ...]
    donor_unit_overlap_ids: tuple[str, ...]
    transported_scientific_object_ids: tuple[str, ...]
    source_substitution_ids: tuple[str, ...]
    collision_audit: ObjectIdentity
    passed: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("audit_id", self.audit_id),
            ("target_slot_id", self.target_slot_id),
            ("threshold_owner_target_slot_id", self.threshold_owner_target_slot_id),
            ("coefficient_owner_target_slot_id", self.coefficient_owner_target_slot_id),
            ("sample_owner_target_slot_id", self.sample_owner_target_slot_id),
            ("likelihood_owner_target_slot_id", self.likelihood_owner_target_slot_id),
            ("action_owner_target_slot_id", self.action_owner_target_slot_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.parent_target_slot_id is not None:
            validate_stable_id(
                self.parent_target_slot_id,
                field_name="parent_target_slot_id",
            )
        for field_name in (
            "shared_unit_target_slot_ids",
            "donor_unit_overlap_ids",
            "transported_scientific_object_ids",
            "source_substitution_ids",
        ):
            require_sorted_unique_strings(getattr(self, field_name), field_name=field_name)
        owners = {
            self.threshold_owner_target_slot_id,
            self.coefficient_owner_target_slot_id,
            self.sample_owner_target_slot_id,
            self.likelihood_owner_target_slot_id,
            self.action_owner_target_slot_id,
        }
        owner_local = owners == {self.target_slot_id}
        nested = self.target_kind is HeldoutTargetKind.NESTED_QUANTUM_NEGATIVE_CONTROL
        if nested:
            if self.parent_target_slot_id is None:
                raise ValueError("nested negative control must name its parent target")
            allowed_shared = (self.parent_target_slot_id,)
            if self.shared_unit_target_slot_ids not in {(), allowed_shared}:
                raise ValueError("nested control may share units only with its frozen parent")
        elif self.parent_target_slot_id is not None or self.shared_unit_target_slot_ids:
            raise ValueError("independent held-out targets cannot share target units")
        expected_pass = (
            owner_local
            and not self.donor_unit_overlap_ids
            and not self.transported_scientific_object_ids
            and not self.source_substitution_ids
        )
        if self.passed != expected_pass or not self.passed:
            raise ValueError("target locality audit detected transport, overlap, or substitution")


@dataclass(frozen=True, slots=True)
class HeldoutTargetDesign(CanonicalRecord):
    """Pre-reveal target-local config, chart, threshold and roster freeze."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/heldout-target-design'

    design_id: str
    target_slot: TargetSlotRecord
    target_kind: HeldoutTargetKind
    compatibility: NativeRoleCompatibilityRecord
    target_config: ObjectIdentity
    source_qualification: ObjectIdentity
    native_threshold_manifest: ObjectIdentity
    native_action_chart: ObjectIdentity
    native_actions: tuple[ObjectIdentity, ...]
    hold_action: ObjectIdentity
    development_roster: ObjectIdentity
    evaluation_roster: ObjectIdentity
    prospective_validation_roster: ObjectIdentity
    native_admission_evaluator: ObjectIdentity
    native_prospective_validation_evaluator: ObjectIdentity
    structural_evaluator: ObjectIdentity
    locality_audit: TargetLocalityAudit
    target_outcomes_accessed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.design_id, field_name="design_id")
        if (
            self.compatibility.target_slot_id != self.target_slot.target_slot_id
            or self.locality_audit.target_slot_id != self.target_slot.target_slot_id
            or self.locality_audit.target_kind is not self.target_kind
        ):
            raise ValueError("target design operands use different target slots")
        if self.compatibility.ontology != self.target_slot.ontology:
            raise ValueError("target design compatibility uses a different ontology")
        require_sorted_unique_ids(
            self.native_actions,
            attribute="object_id",
            field_name="native_actions",
        )
        if self.hold_action.object_id != "hold" or self.hold_action not in self.native_actions:
            raise ValueError("finite native action chart must contain exact hold identity")
        rosters = {
            self.development_roster.object_fingerprint,
            self.evaluation_roster.object_fingerprint,
            self.prospective_validation_roster.object_fingerprint,
        }
        if len(rosters) != 3:
            raise ValueError('development, evaluation, and prospective validation rosters must be disjoint')
        evaluators = {
            self.native_admission_evaluator.object_fingerprint,
            self.native_prospective_validation_evaluator.object_fingerprint,
            self.structural_evaluator.object_fingerprint,
        }
        if len(evaluators) != 3:
            raise ValueError('native admission, native prospective validation, and structural evaluators must be distinct')
        if (
            self.target_kind is HeldoutTargetKind.NESTED_QUANTUM_NEGATIVE_CONTROL
            and self.locality_audit.parent_target_slot_id is None
        ):
            raise ValueError("nested target design lacks a frozen parent")
        if self.target_outcomes_accessed or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("target design must freeze before held-out outcomes")


FOUR_TARGET_STRUCTURAL_RECURRENCE_TARGET_SLOT_IDS = (
    'heldout-quantum-adjacent',
    'nested-quantum-negative-control-global',
    'heldout-quantum-weak',
    'heldout-cross-substrate-fresh',
)

_FOUR_TARGET_STRUCTURAL_RECURRENCE_PROFILE = {
    'heldout-quantum-adjacent': (1, HeldoutTargetKind.HELDOUT_QUANTUM, None),
    'nested-quantum-negative-control-global': (
        2,
        HeldoutTargetKind.NESTED_QUANTUM_NEGATIVE_CONTROL,
        'heldout-quantum-adjacent',
    ),
    'heldout-quantum-weak': (3, HeldoutTargetKind.HELDOUT_QUANTUM, None),
    'heldout-cross-substrate-fresh': (4, HeldoutTargetKind.HELDOUT_CROSS_SUBSTRATE, None),
}


@dataclass(frozen=True, slots=True)
class HeldoutTargetDesignSpec(CanonicalRecord):
    'Caller-supplied identities used to freeze one predeclared held-out structural target slot.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/heldout-target-design-spec'

    spec_id: str
    target_slot_id: str
    target_class_id: str
    evidence_world_id: str
    source: ObjectIdentity
    reserve_source_ids: tuple[str, ...]
    compatibility_bindings: tuple[NativeRoleBinding, ...]
    required_new_role_ids: tuple[str, ...]
    target_config: ObjectIdentity
    source_qualification: ObjectIdentity
    native_threshold_manifest: ObjectIdentity
    native_action_chart: ObjectIdentity
    native_actions: tuple[ObjectIdentity, ...]
    hold_action: ObjectIdentity
    development_roster: ObjectIdentity
    evaluation_roster: ObjectIdentity
    prospective_validation_roster: ObjectIdentity
    native_admission_evaluator: ObjectIdentity
    native_prospective_validation_evaluator: ObjectIdentity
    structural_evaluator: ObjectIdentity
    collision_audit: ObjectIdentity
    donor_unit_overlap_ids: tuple[str, ...]
    transported_scientific_object_ids: tuple[str, ...]
    source_substitution_ids: tuple[str, ...]
    target_outcomes_accessed: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("spec_id", self.spec_id),
            ("target_slot_id", self.target_slot_id),
            ("target_class_id", self.target_class_id),
            ("evidence_world_id", self.evidence_world_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.target_slot_id not in FOUR_TARGET_STRUCTURAL_RECURRENCE_TARGET_SLOT_IDS:
            raise ValueError('target design spec is not one of the four heldout structural slots')
        require_sorted_unique_strings(
            self.reserve_source_ids,
            field_name="reserve_source_ids",
        )
        require_sorted_unique_ids(
            self.compatibility_bindings,
            attribute="binding_id",
            field_name="compatibility_bindings",
        )
        if {binding.role for binding in self.compatibility_bindings} != set(UniversalRole):
            raise ValueError("target design spec must bind all sixteen frozen roles")
        require_sorted_unique_strings(
            self.required_new_role_ids,
            field_name="required_new_role_ids",
        )
        require_sorted_unique_ids(
            self.native_actions,
            attribute="object_id",
            field_name="native_actions",
        )
        if self.hold_action.object_id != "hold" or self.hold_action not in self.native_actions:
            raise ValueError("target design spec must include exact hold action")
        for field_name in (
            "donor_unit_overlap_ids",
            "transported_scientific_object_ids",
            "source_substitution_ids",
        ):
            require_sorted_unique_strings(getattr(self, field_name), field_name=field_name)
        if self.target_outcomes_accessed:
            raise ValueError("held-out design spec contains protected target outcomes")


def build_structural_recurrence_heldout_target_designs(
    *,
    ontology: StructuralOntologyIdentity,
    specs: tuple[HeldoutTargetDesignSpec, ...],
) -> tuple[HeldoutTargetDesign, ...]:
    'Build the exact adjacent, nested-control, weak and cross-substrate roster in canonical eligibility order.'

    require_sorted_unique_ids(specs, attribute="spec_id", field_name="specs")
    by_slot = {spec.target_slot_id: spec for spec in specs}
    if set(by_slot) != set(FOUR_TARGET_STRUCTURAL_RECURRENCE_TARGET_SLOT_IDS) or len(specs) != len(by_slot):
        raise ValueError('structural design factory requires adjacent, nested-control, weak and cross-substrate slots')
    adjacent_quantum = by_slot['heldout-quantum-adjacent']
    nested_quantum_negative_control = by_slot['nested-quantum-negative-control-global']
    if (
        nested_quantum_negative_control.source != adjacent_quantum.source
        or nested_quantum_negative_control.development_roster != adjacent_quantum.development_roster
        or nested_quantum_negative_control.evaluation_roster != adjacent_quantum.evaluation_roster
        or nested_quantum_negative_control.prospective_validation_roster != adjacent_quantum.prospective_validation_roster
        or nested_quantum_negative_control.native_action_chart != adjacent_quantum.native_action_chart
        or nested_quantum_negative_control.native_actions != adjacent_quantum.native_actions
    ):
        raise ValueError('nested quantum negative control must use exact adjacent quantum units and actions')
    independent_sources = {
        by_slot[slot_id].source.canonical_bytes()
        for slot_id in ('heldout-quantum-adjacent', 'heldout-quantum-weak', 'heldout-cross-substrate-fresh')
    }
    if len(independent_sources) != 3:
        raise ValueError('adjacent quantum, weak quantum and cross-substrate slots require distinct frozen sources')
    ontology_identity = ObjectIdentity.from_record(ontology.ontology_id, ontology)
    designs: list[HeldoutTargetDesign] = []
    for slot_id in FOUR_TARGET_STRUCTURAL_RECURRENCE_TARGET_SLOT_IDS:
        spec = by_slot[slot_id]
        order, kind, parent = _FOUR_TARGET_STRUCTURAL_RECURRENCE_PROFILE[slot_id]
        slot = TargetSlotRecord(
            target_slot_id=slot_id,
            target_class_id=spec.target_class_id,
            canonical_order=order,
            target_level=TargetLevel.PROSPECTIVE_USE,
            evidence_world_id=spec.evidence_world_id,
            source=spec.source,
            ontology=ontology_identity,
            reserve_source_ids=spec.reserve_source_ids,
            outcome_naive_at_freeze=True,
            evaluation_outcomes_accessed=False,
        )
        compatibility = NativeRoleCompatibilityRecord(
            map_id=f"{spec.spec_id}.compatibility",
            target_slot_id=slot_id,
            ontology=ontology_identity,
            bindings=spec.compatibility_bindings,
            required_new_role_ids=spec.required_new_role_ids,
            map_valid=not spec.required_new_role_ids,
            reason_codes=(
                () if not spec.required_new_role_ids else ("UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE",)
            ),
        )
        locality = TargetLocalityAudit(
            audit_id=f"{spec.spec_id}.locality",
            target_slot_id=slot_id,
            target_kind=kind,
            parent_target_slot_id=parent,
            threshold_owner_target_slot_id=slot_id,
            coefficient_owner_target_slot_id=slot_id,
            sample_owner_target_slot_id=slot_id,
            likelihood_owner_target_slot_id=slot_id,
            action_owner_target_slot_id=slot_id,
            shared_unit_target_slot_ids=(() if parent is None else (parent,)),
            donor_unit_overlap_ids=spec.donor_unit_overlap_ids,
            transported_scientific_object_ids=spec.transported_scientific_object_ids,
            source_substitution_ids=spec.source_substitution_ids,
            collision_audit=spec.collision_audit,
            passed=not (
                spec.donor_unit_overlap_ids
                or spec.transported_scientific_object_ids
                or spec.source_substitution_ids
            ),
        )
        designs.append(
            HeldoutTargetDesign(
                design_id=f"{spec.spec_id}.design",
                target_slot=slot,
                target_kind=kind,
                compatibility=compatibility,
                target_config=spec.target_config,
                source_qualification=spec.source_qualification,
                native_threshold_manifest=spec.native_threshold_manifest,
                native_action_chart=spec.native_action_chart,
                native_actions=spec.native_actions,
                hold_action=spec.hold_action,
                development_roster=spec.development_roster,
                evaluation_roster=spec.evaluation_roster,
                prospective_validation_roster=spec.prospective_validation_roster,
                native_admission_evaluator=spec.native_admission_evaluator,
                native_prospective_validation_evaluator=spec.native_prospective_validation_evaluator,
                structural_evaluator=spec.structural_evaluator,
                locality_audit=locality,
                target_outcomes_accessed=False,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            )
        )
    return tuple(designs)


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceTruthKnownFixtureResult(CanonicalRecord):
    """One truth-known fixture with primary and all comparator matches."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/methods/structural-recurrence-truth-known-fixture-result'

    fixture_id: str
    fixture_kind: StructuralRecurrenceTruthKnownFixtureKind
    prediction_input: StructuralPredictionInput
    primary_prediction: StructuralPrediction
    comparator_panel: PredictiveComparatorPanel
    observation: StructuralObservation
    matches: tuple[PredictiveMatch, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.fixture_id, field_name="fixture_id")
        input_identity = ObjectIdentity.from_record(
            self.prediction_input.input_id,
            self.prediction_input,
        )
        observation_identity = ObjectIdentity.from_record(
            self.observation.observation_id,
            self.observation,
        )
        if (
            self.primary_prediction.prediction_input != input_identity
            or self.comparator_panel.prediction_input != input_identity
        ):
            raise ValueError("Truth-known fixture predictions do not bind its exact input")
        require_sorted_unique_ids(
            self.matches,
            attribute="match_id",
            field_name="matches",
        )
        if {match.predictor_kind for match in self.matches} != set(PredictorKind):
            raise ValueError("Truth-known fixture requires primary plus all five comparators")
        predictions = {
            self.primary_prediction.predictor_kind: self.primary_prediction,
            **{
                prediction.predictor_kind: prediction
                for prediction in self.comparator_panel.predictions
            },
        }
        for match in self.matches:
            prediction = predictions[match.predictor_kind]
            if (
                match.prediction != ObjectIdentity.from_record(prediction.prediction_id, prediction)
                or match.observation != observation_identity
            ):
                raise ValueError("Truth-known fixture match does not bind exact prediction and observation")
        if not _truth_known_fixture_is_exact(self):
            raise ValueError("Truth-known fixture does not exhibit its frozen planted category")

    @property
    def primary_match(self) -> PredictiveMatch:
        return next(
            match for match in self.matches if match.predictor_kind is PredictorKind.PRIMARY
        )


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceTruthKnownConformanceResult(CanonicalRecord):
    """Complete, order-invariant truth-known gate required before compiler freeze."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/methods/structural-recurrence-truth-known-conformance-result'

    conformance_id: str
    fixtures: tuple[StructuralRecurrenceTruthKnownFixtureResult, ...]
    method_adjudication: PredictiveRecurrenceAdjudication
    false_action_count: int
    hidden_sink_admission_count: int
    safety_singleton_passed: bool
    exact_terminal_categories_passed: bool
    sharper_than_wildcard_passed: bool
    overbroad_comparators_rejected: bool
    qualified: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.conformance_id, field_name="conformance_id")
        require_sorted_unique_ids(
            self.fixtures,
            attribute="fixture_id",
            field_name="fixtures",
        )
        if {fixture.fixture_kind for fixture in self.fixtures} != set(REQUIRED_TRUTH_KNOWN_FIXTURE_KINDS):
            raise ValueError("Truth-known fixture conformance does not contain the complete frozen panel")
        all_matches = tuple(
            sorted(
                (match for fixture in self.fixtures for match in fixture.matches),
                key=lambda value: value.match_id,
            )
        )
        expected_adjudication = PredictiveStructuralAdjudicator().adjudicate(
            adjudication_id=f"{self.conformance_id}.method",
            matches=all_matches,
        )
        if self.method_adjudication != expected_adjudication:
            raise ValueError("Truth-known fixture conformance method adjudication does not bind its exact fixture matches")
        primary = tuple(fixture.primary_match for fixture in self.fixtures)
        expected_false_actions = sum(match.false_action_count for match in primary)
        expected_hidden_sinks = sum(match.hidden_sink_admission_count for match in primary)
        expected_safety = all(
            fixture.primary_prediction.safety.singleton
            and fixture.primary_match.safety_exact
            and fixture.primary_match.reason_exact
            for fixture in self.fixtures
        )
        expected_terminals = all(_truth_known_fixture_is_exact(fixture) for fixture in self.fixtures)
        expected_sharpness = all(
            fixture.primary_prediction.sharpness
            > next(
                prediction.sharpness
                for prediction in fixture.comparator_panel.predictions
                if prediction.predictor_kind is PredictorKind.WILDCARD
            )
            for fixture in self.fixtures
        )
        always_act_rejected = any(
            next(
                match
                for match in fixture.matches
                if match.predictor_kind is PredictorKind.ALWAYS_ACT
            ).false_action_count
            or next(
                match
                for match in fixture.matches
                if match.predictor_kind is PredictorKind.ALWAYS_ACT
            ).hidden_sink_admission_count
            for fixture in self.fixtures
        )
        always_hold_rejected = any(
            next(
                match
                for match in fixture.matches
                if match.predictor_kind is PredictorKind.ALWAYS_HOLD
            ).missed_opportunity_count
            for fixture in self.fixtures
        )
        wildcard_rejected = all(
            next(
                prediction
                for prediction in fixture.comparator_panel.predictions
                if prediction.predictor_kind is PredictorKind.WILDCARD
            ).status
            is PredictionStatus.PREDICTOR_NOT_RESTRICTIVE
            for fixture in self.fixtures
        )
        expected_comparators = always_act_rejected and always_hold_rejected and wildcard_rejected
        expected_qualified = all(
            (
                self.method_adjudication.verdict is MethodQualificationVerdict.METHOD_QUALIFIED,
                expected_false_actions == 0,
                expected_hidden_sinks == 0,
                expected_safety,
                expected_terminals,
                expected_sharpness,
                expected_comparators,
            )
        )
        if (
            self.false_action_count != expected_false_actions
            or self.hidden_sink_admission_count != expected_hidden_sinks
            or self.safety_singleton_passed != expected_safety
            or self.exact_terminal_categories_passed != expected_terminals
            or self.sharper_than_wildcard_passed != expected_sharpness
            or self.overbroad_comparators_rejected != expected_comparators
            or self.qualified != expected_qualified
        ):
            raise ValueError("Truth-known fixture conformance qualification fields are not mechanically derived")
        if not self.qualified:
            raise ValueError("only a fully qualified truth-known fixture conformance result may enter compiler freeze")


def _truth_known_fixture_is_exact(fixture: StructuralRecurrenceTruthKnownFixtureResult) -> bool:
    prediction = fixture.primary_prediction
    observation = fixture.observation
    match = fixture.primary_match
    core = observation.core
    kind = fixture.fixture_kind
    common = (
        prediction.target_slot_id == observation.target_slot_id
        and match.passed
        and not match.false_action_count
        and not match.hidden_sink_admission_count
    )
    if not common:
        return False
    if kind is StructuralRecurrenceTruthKnownFixtureKind.POSITIVE_LOCAL_SINGLETON:
        return (
            core.admission_topology is AdmissionTopology.SINGLETON
            and core.policy_branch is PolicyBranch.EXACT_ACTION
        )
    if kind is StructuralRecurrenceTruthKnownFixtureKind.MULTIPLE_ACTION_DETERMINISTIC_SELECTION:
        return (
            core.admission_topology is AdmissionTopology.MULTIPLE
            and core.policy_branch is PolicyBranch.EXACT_ACTION
            and len(prediction.safety.admission_direction_ids) == 1
        )
    if kind is StructuralRecurrenceTruthKnownFixtureKind.POSITIVE_RESPONSE_HIDDEN_SINK:
        return (
            observation.restrictions.receiver_role is ReceiverRole.SINK_LIMITING
            and core.policy_branch is PolicyBranch.HOLD
            and observation.safety.primary_reason is StructuralReason.RECEIVER_SINK
        )
    if kind is StructuralRecurrenceTruthKnownFixtureKind.EMPTY_ADMISSION:
        return (
            core.admission_topology is AdmissionTopology.EMPTY
            and core.policy_branch is PolicyBranch.HOLD
        )
    if kind is StructuralRecurrenceTruthKnownFixtureKind.MISSING_REALIZED_ACTION:
        return (
            core.policy_branch is PolicyBranch.NONATTEMPT
            and observation.safety.action_realization
            in {ActionRole.ABSENT, ActionRole.MISMATCH, ActionRole.UNEVALUABLE}
            and observation.safety.primary_reason is StructuralReason.ACTION_REALIZATION
        )
    if kind is StructuralRecurrenceTruthKnownFixtureKind.MANDATORY_HOLD:
        return (
            core.admission_topology is AdmissionTopology.HOLD_ONLY
            and core.policy_branch is PolicyBranch.HOLD
        )
    if kind is StructuralRecurrenceTruthKnownFixtureKind.PROSPECTIVE_VALIDATION_SUCCESS:
        return (
            core.policy_branch is PolicyBranch.EXACT_ACTION
            and core.prospective_validation_disposition is ProspectiveDisposition.VALIDATED
        )
    if kind is StructuralRecurrenceTruthKnownFixtureKind.PROSPECTIVE_VALIDATION_OPPOSITION:
        return (
            core.policy_branch is PolicyBranch.EXACT_ACTION
            and core.prospective_validation_disposition is ProspectiveDisposition.OPPOSED
        )
    if kind is StructuralRecurrenceTruthKnownFixtureKind.UNSUPPORTED_DENOMINATOR:
        return (
            core.denominator_structure is DenominatorStructure.INCOMPATIBLE
            and core.policy_branch is PolicyBranch.NONATTEMPT
        )
    if kind is StructuralRecurrenceTruthKnownFixtureKind.LOSSY_COMPATIBILITY_MAP:
        return bool(fixture.prediction_input.missing_measurement_roles) and (
            core.denominator_structure is DenominatorStructure.UNEVALUABLE
            or core.history_clock_quotient is HistoryClockQuotient.UNEVALUABLE
            or core.support_transport is SupportTransport.UNEVALUABLE
            or core.admission_topology is AdmissionTopology.UNEVALUABLE
        )
    return (
        kind is StructuralRecurrenceTruthKnownFixtureKind.NEW_ONTOLOGY_ROLE
        and prediction.status is PredictionStatus.UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE
        and observation.ontology_counterexample
        and match.ontology_counterexample_exact
    )


def evaluate_truth_known_fixture(
    *,
    fixture_id: str,
    fixture_kind: StructuralRecurrenceTruthKnownFixtureKind,
    prediction_input: StructuralPredictionInput,
    observation: StructuralObservation,
) -> StructuralRecurrenceTruthKnownFixtureResult:
    """Compile and match one truth-known case through all frozen comparators."""

    compiler = PredictiveStructuralRecurrenceCompiler()
    matcher = PredictiveStructuralMatcher()
    primary = compiler.compile(prediction_input)
    panel = compiler.comparator_panel(prediction_input)
    matches = tuple(
        sorted(
            (
                matcher.match(primary, observation),
                *(matcher.match(prediction, observation) for prediction in panel.predictions),
            ),
            key=lambda value: value.match_id,
        )
    )
    return StructuralRecurrenceTruthKnownFixtureResult(
        fixture_id=fixture_id,
        fixture_kind=fixture_kind,
        prediction_input=prediction_input,
        primary_prediction=primary,
        comparator_panel=panel,
        observation=observation,
        matches=matches,
    )


def qualify_truth_known_conformance(
    *,
    conformance_id: str,
    fixtures: tuple[StructuralRecurrenceTruthKnownFixtureResult, ...],
) -> StructuralRecurrenceTruthKnownConformanceResult:
    "Reduce the complete planted panel to one deterministic truth-known fixture conformance gate."

    ordered = tuple(sorted(fixtures, key=lambda value: value.fixture_id))
    matches = tuple(
        sorted(
            (match for fixture in ordered for match in fixture.matches),
            key=lambda value: value.match_id,
        )
    )
    adjudication = PredictiveStructuralAdjudicator().adjudicate(
        adjudication_id=f"{conformance_id}.method",
        matches=matches,
    )
    primary = tuple(fixture.primary_match for fixture in ordered)
    false_actions = sum(match.false_action_count for match in primary)
    hidden_sinks = sum(match.hidden_sink_admission_count for match in primary)
    safety = all(
        fixture.primary_prediction.safety.singleton
        and fixture.primary_match.safety_exact
        and fixture.primary_match.reason_exact
        for fixture in ordered
    )
    terminals = all(_truth_known_fixture_is_exact(fixture) for fixture in ordered)
    sharpness = all(
        fixture.primary_prediction.sharpness
        > next(
            prediction.sharpness
            for prediction in fixture.comparator_panel.predictions
            if prediction.predictor_kind is PredictorKind.WILDCARD
        )
        for fixture in ordered
    )
    always_act_rejected = any(
        match.false_action_count or match.hidden_sink_admission_count
        for fixture in ordered
        for match in fixture.matches
        if match.predictor_kind is PredictorKind.ALWAYS_ACT
    )
    always_hold_rejected = any(
        match.missed_opportunity_count
        for fixture in ordered
        for match in fixture.matches
        if match.predictor_kind is PredictorKind.ALWAYS_HOLD
    )
    wildcard_rejected = all(
        prediction.status is PredictionStatus.PREDICTOR_NOT_RESTRICTIVE
        for fixture in ordered
        for prediction in fixture.comparator_panel.predictions
        if prediction.predictor_kind is PredictorKind.WILDCARD
    )
    comparators = always_act_rejected and always_hold_rejected and wildcard_rejected
    qualified = all(
        (
            adjudication.verdict is MethodQualificationVerdict.METHOD_QUALIFIED,
            false_actions == 0,
            hidden_sinks == 0,
            safety,
            terminals,
            sharpness,
            comparators,
        )
    )
    return StructuralRecurrenceTruthKnownConformanceResult(
        conformance_id=conformance_id,
        fixtures=ordered,
        method_adjudication=adjudication,
        false_action_count=false_actions,
        hidden_sink_admission_count=hidden_sinks,
        safety_singleton_passed=safety,
        exact_terminal_categories_passed=terminals,
        sharper_than_wildcard_passed=sharpness,
        overbroad_comparators_rejected=comparators,
        qualified=qualified,
    )


def _planted_truth_known_identity(scope_id: str, object_id: str) -> ObjectIdentity:
    return ObjectIdentity(
        object_id=object_id,
        object_schema='empirical-lawhood/methods/structural-recurrence/synthetic-decision-cell-object',
        object_version="1.0.0",
        object_fingerprint=sha256(f"{scope_id}:{object_id}".encode("ascii")).hexdigest(),
    )


def _build_truth_known_prediction_input(
    *,
    scope_id: str,
    index: int,
    donor_corpus: DonorCorpusIdentity,
    denominator: DenominatorStructure = DenominatorStructure.COMMON_LAW,
    history: HistoryClockQuotient = HistoryClockQuotient.CLOSE_CLOSED,
    support: SupportTransport = SupportTransport.EQUALITY,
    candidate_count: int = 1,
    failed_role: UniversalRole | None = None,
    action_role: ActionRole = ActionRole.REALIZATION_QUALIFIED,
    hold_only_chart: bool = False,
    prospective_validation_expectation: DevelopmentControllerUseExpectation = DevelopmentControllerUseExpectation.VALIDATED,
    missing_roles: tuple[UniversalRole, ...] = (),
    required_new_role_ids: tuple[str, ...] = (),
) -> StructuralPredictionInput:
    ontology = structural_ontology()
    ontology_identity = ObjectIdentity.from_record(ontology.ontology_id, ontology)
    case_id = f"{scope_id}.case.{index:02d}"
    target_slot_id = f"{scope_id}.target.{index:02d}"
    target = TargetSlotRecord(
        target_slot_id=target_slot_id,
        target_class_id=f"truth-known-class.{index:02d}",
        canonical_order=index + 1,
        target_level=TargetLevel.PROSPECTIVE_USE,
        evidence_world_id=f"{scope_id}.world.{index:02d}",
        source=_planted_truth_known_identity(scope_id, f"truth-known-source.{index:02d}"),
        ontology=ontology_identity,
        reserve_source_ids=(),
        outcome_naive_at_freeze=True,
        evaluation_outcomes_accessed=False,
    )
    missing = set(missing_roles)
    bindings = tuple(
        sorted(
            (
                NativeRoleBinding(
                    binding_id=f"{case_id}.binding.{binding_index:02d}",
                    native_object_id=f"{case_id}.native.{binding_index:02d}",
                    role=role,
                    native_unit="1",
                    native_direction="planted",
                    clock_id=f"{case_id}.clock",
                    missingness_rule="Planted absence remains typed.",
                    claim_ceiling=EvidenceCeiling.LOCAL_LAW,
                    available=role not in missing,
                    reuse_compatibility_id=None,
                )
                for binding_index, role in enumerate(UNIVERSAL_ROLES)
            ),
            key=lambda value: value.binding_id,
        )
    )
    compatibility = NativeRoleCompatibilityRecord(
        map_id=f"{case_id}.compatibility",
        target_slot_id=target_slot_id,
        ontology=ontology_identity,
        bindings=bindings,
        required_new_role_ids=required_new_role_ids,
        map_valid=not required_new_role_ids,
        reason_codes=(() if not required_new_role_ids else ("UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE",)),
    )
    actions = tuple(
        ActionCandidateFact(
            action_id=f"{case_id}.action.{action_index:02d}",
            development_rank=candidate_count - action_index,
            target_response_positive=True,
            action_realization=action_role,
            operands=tuple(
                AdmissionOperandFact(
                    operand_id=(f"{case_id}.operand.{action_index:02d}.{operand_index:02d}"),
                    role=role,
                    status=(
                        OperandStatus.ABSENT
                        if role in missing
                        else OperandStatus.FAIL
                        if role is failed_role
                        else OperandStatus.PASS
                    ),
                    evidence=_planted_truth_known_identity(
                        scope_id,
                        (
                            f"truth-known-operand-evidence.{index:02d}."
                            f"{action_index:02d}.{operand_index:02d}"
                        ),
                    ),
                    reason_codes=(
                        ("OPERAND_ABSENT",)
                        if role in missing
                        else ("OPERAND_FAILED",)
                        if role is failed_role
                        else ()
                    ),
                )
                for operand_index, role in enumerate(ADMISSION_ROLES)
            ),
        )
        for action_index in range(candidate_count)
    )
    if failed_role is UniversalRole.RECEIVER_SINK:
        receiver_role = ReceiverRole.SINK_LIMITING
    elif failed_role is UniversalRole.PRESERVATION:
        receiver_role = ReceiverRole.PRESERVATION_LIMITING
    elif failed_role is UniversalRole.RECEIVER_TARGET:
        receiver_role = ReceiverRole.TARGET_LIMITING
    else:
        receiver_role = ReceiverRole.NONLIMITING
    restrictions = StructuralRestrictionProfile(
        preparation_role=PreparationRole.DECISIVE,
        history_role={
            HistoryClockQuotient.CLOSE_CLOSED: HistoryRole.QUOTIENT_ON_DECLARED_PAIR,
            HistoryClockQuotient.CLOSE_DIVERGED: HistoryRole.RETAIN,
            HistoryClockQuotient.TIMING_CONFOUNDED: HistoryRole.TIMING_CONTROL_REQUIRED,
            HistoryClockQuotient.MIXED: HistoryRole.RETAIN,
            HistoryClockQuotient.UNEVALUABLE: HistoryRole.UNEVALUABLE,
        }[history],
        action_role=action_role,
        receiver_role=receiver_role,
        numerical_role=(
            NumericalRole.VIEW_LOCAL
            if denominator is DenominatorStructure.VIEW_LOCAL
            else NumericalRole.QUALIFIED
        ),
        boundary_role=(
            BoundaryRole.SUPPORT_LOSS
            if support is SupportTransport.BOUNDARY_LOSS
            else BoundaryRole.INTERIOR
        ),
        internal_transfer_role=InternalTransferRole.RELEVANT_DIRECTION,
        topology_mode_role=TopologyModeRole.NONDECISIVE_ON_CHART,
    )
    return StructuralPredictionInput(
        input_id=f"{case_id}.input",
        ontology=ontology,
        donor_corpus=donor_corpus,
        target_slot=target,
        compatibility=compatibility,
        denominator_evidence=denominator,
        history_evidence=history,
        support_evidence=support,
        restriction_evidence=restrictions,
        action_candidates=actions,
        hold_only_chart=hold_only_chart,
        deterministic_selector_available=True,
        observer_status=ObserverStatus.QUALIFIED,
        source_ready=True,
        receiver_available=True,
        prospective_units_available=True,
        hold_preservation_expected=True,
        checkpoint_receiver_close=True,
        prospective_validation_expectation=prospective_validation_expectation,
        missing_measurement_roles=tuple(sorted(missing_roles, key=lambda value: value.value)),
        required_new_role_ids=required_new_role_ids,
        secondary_reason_codes=(),
        independent_development_unit_count=8,
        evidence_rung=EvidenceRung.LOCAL_LAW,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        raw_trajectory_values_present=False,
        native_threshold_values_present=False,
        target_admission_outcomes_accessed=False,
        target_prospective_validation_outcomes_accessed=False,
        post_reveal_diagnostics_accessed=False,
    )


def _build_truth_known_observation(
    *,
    scope_id: str,
    index: int,
    prediction_input: StructuralPredictionInput,
    evaluator: ObjectIdentity,
) -> StructuralObservation:
    prediction = PredictiveStructuralRecurrenceCompiler().compile(prediction_input)
    return StructuralObservation(
        observation_id=f"{scope_id}.observation.{index:02d}",
        target_slot_id=prediction.target_slot_id,
        target_level=prediction.target_level,
        evidence_world_id=prediction.evidence_world_id,
        target_handoff=_planted_truth_known_identity(
            scope_id,
            f"truth-known-target-handoff.{index:02d}",
        ),
        evaluator=evaluator,
        core=StructuralObservedCore(
            denominator_structure=prediction.core.denominator_structure[0],
            history_clock_quotient=prediction.core.history_clock_quotient[0],
            support_transport=prediction.core.support_transport[0],
            admission_topology=prediction.core.admission_topology[0],
            policy_branch=prediction.core.policy_branch[0],
            prospective_validation_disposition=prediction.core.prospective_validation_disposition[0],
        ),
        restrictions=prediction.restrictions,
        safety=StructuralSafetyObservation(
            policy_branch=prediction.safety.policy_branch[0],
            primary_reason=prediction.safety.primary_reason[0],
            action_realization=prediction.safety.action_realization[0],
            hold_preservation=prediction.safety.hold_preservation[0],
            prospective_validation_condition_false=prediction.safety.prospective_validation_condition_false[0],
            admission_direction_id=prediction.safety.admission_direction_ids[0],
        ),
        ontology_counterexample=(
            prediction.status is PredictionStatus.UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE
        ),
        required_new_role_ids=prediction.required_new_role_ids,
        independent_unit_count=16,
        observed_rung=EvidenceRung.CONTROLLER_USE,
        outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
    )


def build_truth_known_conformance(
    *,
    conformance_id: str,
    truth_known_handoff: ObjectIdentity,
    truth_known_evaluator: ObjectIdentity,
    decision_table_id: str,
    decision_table_sha256: str,
) -> StructuralRecurrenceTruthKnownConformanceResult:
    """Build and adjudicate the complete frozen software-conformance panel."""

    validate_stable_id(conformance_id, field_name="conformance_id")
    validate_stable_id(decision_table_id, field_name="decision_table_id")
    validate_sha256(decision_table_sha256, field_name="decision_table_sha256")
    target_slot_ids = tuple(
        f"{conformance_id}.target.{index:02d}" for index in range(len(REQUIRED_TRUTH_KNOWN_FIXTURE_KINDS))
    )
    ontology = structural_ontology()
    donor_corpus = DonorCorpusIdentity(
        corpus_id=f"{conformance_id}.donor",
        ontology=ObjectIdentity.from_record(ontology.ontology_id, ontology),
        donor_handoffs=(truth_known_handoff,),
        evidence_cutoff_rung=EvidenceRung.LOCAL_LAW,
        evidence_cutoff_id=f"{conformance_id}.local-law-cutoff",
        decision_table_id=decision_table_id,
        decision_table_sha256=decision_table_sha256,
        excluded_target_slot_ids=tuple(sorted(target_slot_ids)),
    )
    definitions: tuple[
        tuple[StructuralRecurrenceTruthKnownFixtureKind, dict[str, Any]],
        ...,
    ] = (
        (StructuralRecurrenceTruthKnownFixtureKind.POSITIVE_LOCAL_SINGLETON, {}),
        (
            StructuralRecurrenceTruthKnownFixtureKind.MULTIPLE_ACTION_DETERMINISTIC_SELECTION,
            {"candidate_count": 2},
        ),
        (
            StructuralRecurrenceTruthKnownFixtureKind.POSITIVE_RESPONSE_HIDDEN_SINK,
            {"failed_role": UniversalRole.RECEIVER_SINK},
        ),
        (StructuralRecurrenceTruthKnownFixtureKind.EMPTY_ADMISSION, {"failed_role": UniversalRole.RECEIVER_TARGET}),
        (
            StructuralRecurrenceTruthKnownFixtureKind.MISSING_REALIZED_ACTION,
            {"action_role": ActionRole.ABSENT},
        ),
        (
            StructuralRecurrenceTruthKnownFixtureKind.MANDATORY_HOLD,
            {"candidate_count": 0, "hold_only_chart": True},
        ),
        (StructuralRecurrenceTruthKnownFixtureKind.PROSPECTIVE_VALIDATION_SUCCESS, {}),
        (
            StructuralRecurrenceTruthKnownFixtureKind.PROSPECTIVE_VALIDATION_OPPOSITION,
            {'prospective_validation_expectation': DevelopmentControllerUseExpectation.OPPOSED},
        ),
        (
            StructuralRecurrenceTruthKnownFixtureKind.UNSUPPORTED_DENOMINATOR,
            {"denominator": DenominatorStructure.INCOMPATIBLE},
        ),
        (
            StructuralRecurrenceTruthKnownFixtureKind.LOSSY_COMPATIBILITY_MAP,
            {
                "support": SupportTransport.UNEVALUABLE,
                "missing_roles": (UniversalRole.SUPPORT,),
            },
        ),
        (
            StructuralRecurrenceTruthKnownFixtureKind.NEW_ONTOLOGY_ROLE,
            {"required_new_role_ids": ("planted-new-core-role",)},
        ),
    )
    fixtures: list[StructuralRecurrenceTruthKnownFixtureResult] = []
    for index, (fixture_kind, values) in enumerate(definitions):
        prediction_input = _build_truth_known_prediction_input(
            scope_id=conformance_id,
            index=index,
            donor_corpus=donor_corpus,
            denominator=values.get(
                "denominator",
                DenominatorStructure.COMMON_LAW,
            ),
            support=values.get(
                "support",
                SupportTransport.EQUALITY,
            ),
            candidate_count=values.get("candidate_count", 1),
            failed_role=values.get("failed_role"),
            action_role=values.get(
                "action_role",
                ActionRole.REALIZATION_QUALIFIED,
            ),
            hold_only_chart=values.get("hold_only_chart", False),
            prospective_validation_expectation=values.get(
                'prospective_validation_expectation',
                DevelopmentControllerUseExpectation.VALIDATED,
            ),
            missing_roles=values.get("missing_roles", ()),
            required_new_role_ids=values.get(
                "required_new_role_ids",
                (),
            ),
        )
        fixtures.append(
            evaluate_truth_known_fixture(
                fixture_id=f"{conformance_id}.fixture.{index:02d}",
                fixture_kind=fixture_kind,
                prediction_input=prediction_input,
                observation=_build_truth_known_observation(
                    scope_id=conformance_id,
                    index=index,
                    prediction_input=prediction_input,
                    evaluator=truth_known_evaluator,
                ),
            )
        )
    return qualify_truth_known_conformance(
        conformance_id=conformance_id,
        fixtures=tuple(fixtures),
    )


COMPILER_INPUT_SCHEMA_IDS = (StructuralPredictionInput.SCHEMA,)
COMPILER_OUTPUT_SCHEMA_IDS = tuple(
    sorted(
        (
            PredictiveComparatorPanel.SCHEMA,
            StructuralPrediction.SCHEMA,
        )
    )
)


@dataclass(frozen=True, slots=True)
class StructuralCompilerFreeze(CanonicalRecord):
    "Immutable structural compiler inputs; publishing the record remains an external act."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-compiler-freeze'

    freeze_id: str
    plan: ObjectIdentity
    ontology: StructuralOntologyIdentity
    donor_corpus: StructuralDonorCorpus
    prediction_donor_identity: DonorCorpusIdentity
    compiler_source_closure: tuple[ObjectIdentity, ...]
    matcher_source_closure: tuple[ObjectIdentity, ...]
    evaluator_source_closure: tuple[ObjectIdentity, ...]
    compiler_implementation_sha256: str
    matcher_implementation_sha256: str
    evaluator_implementation_sha256: str
    ontology_codebook: ObjectIdentity
    allowed_input_schema_ids: tuple[str, ...]
    allowed_output_schema_ids: tuple[str, ...]
    truth_known_conformance: StructuralRecurrenceTruthKnownConformanceResult
    target_designs: tuple[HeldoutTargetDesign, ...]
    common_target_design_freeze: ObjectIdentity
    prediction_rule: ObjectIdentity
    comparison_rule: ObjectIdentity
    terminal_verdict_rule: ObjectIdentity
    forbidden_claim_ids: tuple[str, ...]
    target_evaluation_outcomes_accessed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        ontology_identity = ObjectIdentity.from_record(
            self.ontology.ontology_id,
            self.ontology,
        )
        if (
            self.donor_corpus.ontology != ontology_identity
            or self.prediction_donor_identity.ontology != ontology_identity
        ):
            raise ValueError("compiler freeze operands use different ontology bytes")
        if (
            self.prediction_donor_identity.donor_handoffs != self.donor_corpus.controlling_handoffs
            or self.prediction_donor_identity.evidence_cutoff_rung
            is not self.donor_corpus.target_prediction_input_cutoff_rung
        ):
            raise ValueError(
                "prediction donor identity must bind donor handoffs and the separate law qualification cutoff"
            )
        for field_name in (
            "compiler_source_closure",
            "matcher_source_closure",
            "evaluator_source_closure",
        ):
            values = getattr(self, field_name)
            require_sorted_unique_ids(
                values,
                attribute="object_id",
                field_name=field_name,
            )
            if not values:
                raise ValueError(f"{field_name} must not be empty")
        for field_name in (
            "compiler_implementation_sha256",
            "matcher_implementation_sha256",
            "evaluator_implementation_sha256",
        ):
            validate_sha256(getattr(self, field_name), field_name=field_name)
        if (
            len(
                {
                    self.compiler_implementation_sha256,
                    self.matcher_implementation_sha256,
                    self.evaluator_implementation_sha256,
                }
            )
            != 3
        ):
            raise ValueError("predictor, matcher, and evaluator fingerprints must be distinct")
        require_sorted_unique_strings(
            self.allowed_input_schema_ids,
            field_name="allowed_input_schema_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.allowed_output_schema_ids,
            field_name="allowed_output_schema_ids",
            allow_empty=False,
        )
        if self.allowed_input_schema_ids != COMPILER_INPUT_SCHEMA_IDS:
            raise ValueError('compiler input schema set differs from the frozen structural recurrence input')
        if self.allowed_output_schema_ids != COMPILER_OUTPUT_SCHEMA_IDS:
            raise ValueError('compiler output schema set differs from the frozen structural recurrence outputs')
        if not self.truth_known_conformance.qualified:
            raise ValueError("compiler cannot freeze before complete truth-known fixture conformance qualification")
        if any(
            fixture.primary_prediction.ontology != ontology_identity
            for fixture in self.truth_known_conformance.fixtures
        ):
            raise ValueError("Truth-known fixture conformance uses a different frozen ontology")
        ordered_designs = tuple(
            sorted(
                self.target_designs,
                key=lambda value: (
                    value.target_slot.canonical_order,
                    value.target_slot.target_slot_id,
                ),
            )
        )
        if self.target_designs != ordered_designs or not self.target_designs:
            raise ValueError("target designs must retain nonempty canonical eligibility order")
        slot_ids = tuple(design.target_slot.target_slot_id for design in self.target_designs)
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("compiler freeze contains duplicate target slots")
        if tuple(design.target_slot.canonical_order for design in self.target_designs) != tuple(
            range(1, len(self.target_designs) + 1)
        ):
            raise ValueError("target canonical order must be contiguous")
        if self.donor_corpus.excluded_target_slot_ids != tuple(sorted(slot_ids)):
            raise ValueError("donor corpus target exclusions differ from frozen target roster")
        if self.prediction_donor_identity.excluded_target_slot_ids != tuple(sorted(slot_ids)):
            raise ValueError("prediction donor exclusions differ from frozen target roster")
        if any(design.target_slot.ontology != ontology_identity for design in self.target_designs):
            raise ValueError("target design uses a different frozen ontology")
        kinds = {
            design.target_kind
            for design in self.target_designs
            if design.target_kind is not HeldoutTargetKind.NESTED_QUANTUM_NEGATIVE_CONTROL
        }
        if (
            not {
                HeldoutTargetKind.HELDOUT_QUANTUM,
                HeldoutTargetKind.HELDOUT_CROSS_SUBSTRATE,
            }
            <= kinds
        ):
            raise ValueError(
                "held-out roster requires quantum and cross-substrate independent targets"
            )
        by_slot = {design.target_slot.target_slot_id: design for design in self.target_designs}
        for design in self.target_designs:
            parent = design.locality_audit.parent_target_slot_id
            if parent is not None:
                if (
                    parent not in by_slot
                    or by_slot[parent].target_kind is not HeldoutTargetKind.HELDOUT_QUANTUM
                ):
                    raise ValueError("nested negative-control parent is not frozen quantum")
        require_sorted_unique_strings(
            self.forbidden_claim_ids,
            field_name="forbidden_claim_ids",
            allow_empty=False,
        )
        if (
            self.target_evaluation_outcomes_accessed
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("structural compiler freeze must predate every held-out evaluation outcome")


def make_prediction_donor_identity(
    *,
    corpus: StructuralDonorCorpus,
    decision_table_id: str,
    decision_table_sha256: str,
) -> DonorCorpusIdentity:
    "Lower the admission/prospective validation-visible corpus to the target-input identity capped at law qualification."

    return DonorCorpusIdentity(
        corpus_id=f"{corpus.corpus_id}.prediction-input",
        ontology=corpus.ontology,
        donor_handoffs=corpus.controlling_handoffs,
        evidence_cutoff_rung=corpus.target_prediction_input_cutoff_rung,
        evidence_cutoff_id=f"{corpus.corpus_id}.target-law-qualification-cutoff",
        decision_table_id=decision_table_id,
        decision_table_sha256=decision_table_sha256,
        excluded_target_slot_ids=corpus.excluded_target_slot_ids,
    )


FOUR_TARGET_STRUCTURAL_RECURRENCE_FORBIDDEN_CLAIM_IDS = (
    "coefficient-universality",
    "literal-universality",
    "transported-native-control",
    'validated-controller-use-without-native-admission-and-validation',
)


def build_structural_recurrence_compiler_freeze(
    *,
    freeze_id: str,
    plan: ObjectIdentity,
    donor_corpus: StructuralDonorCorpus,
    decision_table_id: str,
    decision_table_sha256: str,
    compiler_source_closure: tuple[ObjectIdentity, ...],
    matcher_source_closure: tuple[ObjectIdentity, ...],
    evaluator_source_closure: tuple[ObjectIdentity, ...],
    compiler_implementation_sha256: str,
    matcher_implementation_sha256: str,
    evaluator_implementation_sha256: str,
    ontology_codebook: ObjectIdentity,
    truth_known_conformance: StructuralRecurrenceTruthKnownConformanceResult,
    target_designs: tuple[HeldoutTargetDesign, ...],
    common_target_design_freeze: ObjectIdentity,
    prediction_rule: ObjectIdentity,
    comparison_rule: ObjectIdentity,
    terminal_verdict_rule: ObjectIdentity,
) -> StructuralCompilerFreeze:
    'Construct exact structural counterfactual inputs for the four-slot held-out roster.'

    if (
        tuple(design.target_slot.target_slot_id for design in target_designs)
        != FOUR_TARGET_STRUCTURAL_RECURRENCE_TARGET_SLOT_IDS
    ):
        raise ValueError('structural law study target designs differ from the four frozen slots and order')
    for design in target_designs:
        order, kind, parent = _FOUR_TARGET_STRUCTURAL_RECURRENCE_PROFILE[design.target_slot.target_slot_id]
        if (
            design.target_slot.canonical_order != order
            or design.target_kind is not kind
            or design.locality_audit.parent_target_slot_id != parent
        ):
            raise ValueError('structural law study target kind, parent, or order differs from plan')
    if donor_corpus.excluded_target_slot_ids != tuple(sorted(FOUR_TARGET_STRUCTURAL_RECURRENCE_TARGET_SLOT_IDS)):
        raise ValueError('structural donor corpus must exclude all four heldout target slots')
    prediction_donor_identity = make_prediction_donor_identity(
        corpus=donor_corpus,
        decision_table_id=decision_table_id,
        decision_table_sha256=decision_table_sha256,
    )
    return StructuralCompilerFreeze(
        freeze_id=freeze_id,
        plan=plan,
        ontology=structural_ontology(),
        donor_corpus=donor_corpus,
        prediction_donor_identity=prediction_donor_identity,
        compiler_source_closure=compiler_source_closure,
        matcher_source_closure=matcher_source_closure,
        evaluator_source_closure=evaluator_source_closure,
        compiler_implementation_sha256=compiler_implementation_sha256,
        matcher_implementation_sha256=matcher_implementation_sha256,
        evaluator_implementation_sha256=evaluator_implementation_sha256,
        ontology_codebook=ontology_codebook,
        allowed_input_schema_ids=COMPILER_INPUT_SCHEMA_IDS,
        allowed_output_schema_ids=COMPILER_OUTPUT_SCHEMA_IDS,
        truth_known_conformance=truth_known_conformance,
        target_designs=target_designs,
        common_target_design_freeze=common_target_design_freeze,
        prediction_rule=prediction_rule,
        comparison_rule=comparison_rule,
        terminal_verdict_rule=terminal_verdict_rule,
        forbidden_claim_ids=FOUR_TARGET_STRUCTURAL_RECURRENCE_FORBIDDEN_CLAIM_IDS,
        target_evaluation_outcomes_accessed=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


@dataclass(frozen=True, slots=True)
class StructuralPredictionIssue(CanonicalRecord):
    """Exact outcome-blind primary/comparator package committed before reveal."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-prediction-issue'

    issue_id: str
    compiler_freeze: StructuralCompilerFreeze
    target_design: HeldoutTargetDesign
    comparison_level: TargetLevel
    prediction_input: StructuralPredictionInput
    primary_prediction: StructuralPrediction
    comparator_panel: PredictiveComparatorPanel
    evaluator: ObjectIdentity
    prediction_commitment: ObjectIdentity
    issued_before_target_outcomes: bool
    target_admission_outcomes_accessed: bool
    target_prospective_validation_outcomes_accessed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.issue_id, field_name="issue_id")
        frozen_design = next(
            (
                design
                for design in self.compiler_freeze.target_designs
                if design.target_slot.target_slot_id
                == self.target_design.target_slot.target_slot_id
            ),
            None,
        )
        if frozen_design != self.target_design:
            raise ValueError("prediction issue target design differs from structural compiler freeze")
        if (
            _TARGET_LEVEL_INDEX[self.comparison_level]
            > _TARGET_LEVEL_INDEX[self.target_design.target_slot.target_level]
        ):
            raise ValueError("prediction issue level exceeds the frozen target maximum")
        expected_slot = replace(
            self.target_design.target_slot,
            target_level=self.comparison_level,
        )
        if (
            self.prediction_input.ontology != self.compiler_freeze.ontology
            or self.prediction_input.donor_corpus != self.compiler_freeze.prediction_donor_identity
            or self.prediction_input.target_slot != expected_slot
            or self.prediction_input.compatibility != self.target_design.compatibility
            or self.primary_prediction.target_level is not self.comparison_level
        ):
            raise ValueError("prediction input differs from exact compiler/target freeze")
        expected_action_ids = {
            action.object_id
            for action in self.target_design.native_actions
            if action != self.target_design.hold_action
        }
        observed_action_ids = {
            candidate.action_id for candidate in self.prediction_input.action_candidates
        }
        if self.prediction_input.hold_only_chart:
            expected_action_ids = set()
        if observed_action_ids != expected_action_ids:
            raise ValueError("prediction action candidates differ from frozen native action bytes")
        expected_primary = PredictiveStructuralRecurrenceCompiler().compile(self.prediction_input)
        expected_panel = PredictiveStructuralRecurrenceCompiler().comparator_panel(
            self.prediction_input
        )
        if self.primary_prediction != expected_primary or self.comparator_panel != expected_panel:
            raise ValueError("prediction issue bytes differ from deterministic compiler output")
        if self.primary_prediction.status is PredictionStatus.PREDICTOR_NOT_RESTRICTIVE:
            raise ValueError("nonrestrictive primary output cannot be issued")
        if self.evaluator != self.target_design.structural_evaluator:
            raise ValueError("prediction issue evaluator differs from target design")
        if (
            not self.issued_before_target_outcomes
            or self.target_admission_outcomes_accessed
            or self.target_prospective_validation_outcomes_accessed
            or self.outcome_access
            not in {OutcomeAccess.OUTCOME_BLIND, OutcomeAccess.DEVELOPMENT_VISIBLE}
        ):
            raise ValueError("prediction issue contains or follows protected target outcomes")


def issue_structural_prediction(
    *,
    issue_id: str,
    compiler_freeze: StructuralCompilerFreeze,
    target_design: HeldoutTargetDesign,
    prediction_input: StructuralPredictionInput,
    prediction_commitment: ObjectIdentity,
) -> StructuralPredictionIssue:
    """Create a pure pre-reveal issue; publication/custody stays outside this method."""

    compiler = PredictiveStructuralRecurrenceCompiler()
    return StructuralPredictionIssue(
        issue_id=issue_id,
        compiler_freeze=compiler_freeze,
        target_design=target_design,
        comparison_level=prediction_input.target_slot.target_level,
        prediction_input=prediction_input,
        primary_prediction=compiler.compile(prediction_input),
        comparator_panel=compiler.comparator_panel(prediction_input),
        evaluator=target_design.structural_evaluator,
        prediction_commitment=prediction_commitment,
        issued_before_target_outcomes=True,
        target_admission_outcomes_accessed=False,
        target_prospective_validation_outcomes_accessed=False,
        outcome_access=prediction_input.outcome_access,
    )


FOUR_TARGET_STRUCTURAL_RECURRENCE_LEVEL_BUNDLE_LEVELS = (
    TargetLevel.MEASUREMENT_READINESS,
    TargetLevel.LAW_QUALIFICATION,
    TargetLevel.ADMISSION,
    TargetLevel.PROSPECTIVE_USE,
)


@dataclass(frozen=True, slots=True)
class StructuralPredictionLevelBundle(CanonicalRecord):
    'measurement readiness--prospective use predictions jointly committed before any target evaluation reveal.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-prediction-level-bundle'

    bundle_id: str
    compiler_freeze: StructuralCompilerFreeze
    target_design: HeldoutTargetDesign
    level_issues: tuple[StructuralPredictionIssue, ...]
    bundle_commitment: ObjectIdentity
    issued_before_target_outcomes: bool
    target_evaluation_outcomes_accessed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        if self.target_design.target_slot.target_level is not TargetLevel.PROSPECTIVE_USE:
            raise ValueError('structural level bundle requires a frozen prospective use target maximum')
        if tuple(issue.comparison_level for issue in self.level_issues) != (
            FOUR_TARGET_STRUCTURAL_RECURRENCE_LEVEL_BUNDLE_LEVELS
        ):
            raise ValueError('prediction bundle must contain exact ordered measurement readiness--prospective use issues')
        if any(
            issue.compiler_freeze != self.compiler_freeze
            or issue.target_design != self.target_design
            for issue in self.level_issues
        ):
            raise ValueError("prediction bundle issues use different freeze or target bytes")
        issue_commitments = tuple(issue.prediction_commitment for issue in self.level_issues)
        if (
            len({value.canonical_bytes() for value in issue_commitments})
            != len(FOUR_TARGET_STRUCTURAL_RECURRENCE_LEVEL_BUNDLE_LEVELS)
            or self.bundle_commitment in issue_commitments
        ):
            raise ValueError('bundle and measurement readiness--prospective use commitments must be distinct')
        normalized_inputs = tuple(
            replace(
                issue.prediction_input,
                input_id=f"{self.bundle_id}.normalized-input",
                target_slot=self.target_design.target_slot,
            )
            for issue in self.level_issues
        )
        if len(set(value.canonical_bytes() for value in normalized_inputs)) != 1:
            raise ValueError("level bundle changes scientific input across target levels")
        if (
            not self.issued_before_target_outcomes
            or self.target_evaluation_outcomes_accessed
            or self.outcome_access
            not in {OutcomeAccess.OUTCOME_BLIND, OutcomeAccess.DEVELOPMENT_VISIBLE}
        ):
            raise ValueError("prediction level bundle contains or follows target outcomes")

    def issue_for(self, achieved_level: TargetLevel) -> StructuralPredictionIssue:
        """Return an already committed level issue without compiling after reveal."""

        for issue in self.level_issues:
            if issue.comparison_level is achieved_level:
                return issue
        raise ValueError('achieved level is outside the committed measurement readiness--prospective use bundle')


def issue_structural_prediction_level_bundle(
    *,
    bundle_id: str,
    compiler_freeze: StructuralCompilerFreeze,
    target_design: HeldoutTargetDesign,
    prediction_input: StructuralPredictionInput,
    level_commitments: tuple[ObjectIdentity, ...],
    bundle_commitment: ObjectIdentity,
) -> StructuralPredictionLevelBundle:
    'Compile exact measurement readiness--prospective use shadows in one outcome-blind prospective act.'

    if prediction_input.target_slot != target_design.target_slot:
        raise ValueError("bundle base input must use the frozen maximum-level target slot")
    if len(level_commitments) != len(FOUR_TARGET_STRUCTURAL_RECURRENCE_LEVEL_BUNDLE_LEVELS):
        raise ValueError('one immutable commitment is required for each measurement readiness--prospective use issue')
    issues = tuple(
        issue_structural_prediction(
            issue_id=f"{bundle_id}.{level.value.lower()}",
            compiler_freeze=compiler_freeze,
            target_design=target_design,
            prediction_input=replace(
                prediction_input,
                input_id=f"{prediction_input.input_id}.{level.value.lower()}",
                target_slot=replace(
                    prediction_input.target_slot,
                    target_level=level,
                ),
            ),
            prediction_commitment=commitment,
        )
        for level, commitment in zip(
            FOUR_TARGET_STRUCTURAL_RECURRENCE_LEVEL_BUNDLE_LEVELS,
            level_commitments,
            strict=True,
        )
    )
    return StructuralPredictionLevelBundle(
        bundle_id=bundle_id,
        compiler_freeze=compiler_freeze,
        target_design=target_design,
        level_issues=issues,
        bundle_commitment=bundle_commitment,
        issued_before_target_outcomes=True,
        target_evaluation_outcomes_accessed=False,
        outcome_access=prediction_input.outcome_access,
    )


def select_issued_prediction_for_achieved_level(
    bundle: StructuralPredictionLevelBundle,
    *,
    achieved_level: TargetLevel,
) -> StructuralPredictionIssue:
    """Select, but never derive, the prediction committed for an achieved level."""

    return bundle.issue_for(achieved_level)


@dataclass(frozen=True, slots=True)
class NativeHeldoutTargetEvidence(CanonicalRecord):
    'Target-owned terminal chain; structural records cannot replace native admission/prospective validation.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/native-heldout-target-evidence'

    evidence_id: str
    target_design: HeldoutTargetDesign
    achieved_level: TargetLevel
    evidence_world_id: str
    source: ObjectIdentity
    target_handoff: ObjectIdentity
    law_qualification_handoff: ObjectIdentity | None
    admission_handoff: ObjectIdentity | None
    prospective_validation_issue: ObjectIdentity | None
    prospective_validation_authority: ObjectIdentity | None
    prospective_validation_handoff: ObjectIdentity | None
    native_admission_topology: AdmissionTopology | None
    native_admission_operands: tuple[AdmissionOperandFact, ...]
    selected_native_action: ObjectIdentity | None
    native_prospective_validation_terminal: NativeControllerUseTerminal | None
    requested_action: ObjectIdentity | None
    accepted_action: ObjectIdentity | None
    applied_action: ObjectIdentity | None
    realized_action: ObjectIdentity | None
    development_roster: ObjectIdentity
    evaluation_roster: ObjectIdentity
    prospective_validation_roster: ObjectIdentity
    native_threshold_manifest: ObjectIdentity
    native_action_chart: ObjectIdentity
    law_qualification_independent_unit_count: int
    admission_independent_unit_count: int
    prospective_validation_independent_unit_count: int
    structural_prediction_used_as_native_operand: bool
    transported_scientific_object_ids: tuple[str, ...]
    source_substitution_ids: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        validate_stable_id(self.evidence_world_id, field_name="evidence_world_id")
        design = self.target_design
        if self.source != design.target_slot.source:
            raise ValueError("held-out evidence source differs from frozen target source")
        if (
            self.development_roster != design.development_roster
            or self.evaluation_roster != design.evaluation_roster
            or self.prospective_validation_roster != design.prospective_validation_roster
            or self.native_threshold_manifest != design.native_threshold_manifest
            or self.native_action_chart != design.native_action_chart
        ):
            raise ValueError("held-out evidence substituted a frozen roster or native object")
        if (
            _TARGET_LEVEL_INDEX[self.achieved_level]
            > _TARGET_LEVEL_INDEX[design.target_slot.target_level]
        ):
            raise ValueError("achieved level exceeds the frozen target maximum")
        for field_name in (
            "transported_scientific_object_ids",
            "source_substitution_ids",
        ):
            require_sorted_unique_strings(getattr(self, field_name), field_name=field_name)
        if (
            self.structural_prediction_used_as_native_operand
            or self.transported_scientific_object_ids
            or self.source_substitution_ids
        ):
            raise ValueError("native target evidence contains transport or substitution")
        if self.outcome_access not in {
            OutcomeAccess.EVALUATOR_REVEAL,
            OutcomeAccess.EVALUATION_REVEALED,
            OutcomeAccess.PRIVILEGED_TRUTH,
        }:
            raise ValueError("terminal target evidence requires evaluator-scoped outcome access")
        for name, value in (
            ('law_qualification_independent_unit_count', self.law_qualification_independent_unit_count),
            ('admission_independent_unit_count', self.admission_independent_unit_count),
            ('prospective_validation_independent_unit_count', self.prospective_validation_independent_unit_count),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        level = _TARGET_LEVEL_INDEX[self.achieved_level]
        law_qualification_required = level >= _TARGET_LEVEL_INDEX[TargetLevel.LAW_QUALIFICATION]
        admission_required = level >= _TARGET_LEVEL_INDEX[TargetLevel.ADMISSION]
        prospective_validation_required = level >= _TARGET_LEVEL_INDEX[TargetLevel.PROSPECTIVE_USE]
        if law_qualification_required != (self.law_qualification_handoff is not None) or law_qualification_required != (
            self.law_qualification_independent_unit_count > 0
        ):
            raise ValueError("achieved level and native law qualification handoff/count differ")
        if admission_required != (self.admission_handoff is not None) or admission_required != (
            self.native_admission_topology is not None
        ):
            raise ValueError('achieved level and native admission handoff/topology differ')
        if admission_required:
            if self.admission_independent_unit_count < 1:
                raise ValueError('native admission requires independent evaluation units')
            if tuple(operand.role for operand in self.native_admission_operands) != ADMISSION_ROLES:
                raise ValueError('native admission must retain the exact ten-role intersection')
            require_sorted_unique_ids(
                self.native_admission_operands,
                attribute="operand_id",
                field_name='native_admission_operands',
            )
        elif self.native_admission_operands or self.admission_independent_unit_count:
            raise ValueError('sub-admission evidence cannot contain native admission operands')
        prospective_validation_objects = (
            self.prospective_validation_issue,
            self.prospective_validation_authority,
            self.prospective_validation_handoff,
            self.native_prospective_validation_terminal,
        )
        if prospective_validation_required != all(value is not None for value in prospective_validation_objects):
            raise ValueError('prospective use/controller use ceiling requires separate prospective validation issue, authority, handoff, and terminal')
        if not prospective_validation_required and (
            any(value is not None for value in prospective_validation_objects) or self.prospective_validation_independent_unit_count
        ):
            raise ValueError('sub-prospective use evidence cannot contain prospective validation outcomes')
        if self.selected_native_action is not None:
            if (
                self.selected_native_action not in design.native_actions
                or self.selected_native_action == design.hold_action
            ):
                raise ValueError("selected action is not an exact frozen non-hold action")
        topology = self.native_admission_topology
        nonhold_admission = topology in {
            AdmissionTopology.SINGLETON,
            AdmissionTopology.MULTIPLE,
        }
        if admission_required and nonhold_admission:
            if self.selected_native_action is None or any(
                operand.status is not OperandStatus.PASS for operand in self.native_admission_operands
            ):
                raise ValueError(
                    'nonempty native admission needs exact selected action and all ten operands'
                )
        elif self.selected_native_action is not None:
            raise ValueError('empty/hold/unevaluable admission cannot select a non-hold action')
        if topology is AdmissionTopology.EMPTY and all(
            operand.status is OperandStatus.PASS for operand in self.native_admission_operands
        ):
            raise ValueError('empty native admission requires at least one failed operand')
        if topology is AdmissionTopology.UNEVALUABLE and not any(
            operand.status in {OperandStatus.ABSENT, OperandStatus.UNEVALUABLE}
            for operand in self.native_admission_operands
        ):
            raise ValueError('unevaluable native admission requires an absent/unevaluable operand')
        terminal = self.native_prospective_validation_terminal
        action_terminal = terminal in {
            NativeControllerUseTerminal.VALIDATED,
            NativeControllerUseTerminal.OPPOSED,
        }
        action_chain = (
            self.requested_action,
            self.accepted_action,
            self.applied_action,
            self.realized_action,
        )
        if action_terminal:
            if not nonhold_admission or not all(value is not None for value in action_chain):
                raise ValueError(
                    'native prospective validation validation/opposition requires prior admission and full action chain'
                )
            if self.prospective_validation_independent_unit_count < 1:
                raise ValueError('native prospective validation action result requires fresh independent units')
        elif not prospective_validation_required and any(value is not None for value in action_chain):
            raise ValueError('sub-prospective use evidence cannot contain prospective validation action records')
        present_action_ids = tuple(value.object_id for value in action_chain if value is not None)
        if len(set(present_action_ids)) != len(present_action_ids):
            raise ValueError("requested, accepted, applied, and realized records stay distinct")


def _structural_prospective_from_native(value: NativeControllerUseTerminal) -> ProspectiveDisposition:
    return {
        NativeControllerUseTerminal.VALIDATED: ProspectiveDisposition.VALIDATED,
        NativeControllerUseTerminal.OPPOSED: ProspectiveDisposition.OPPOSED,
        NativeControllerUseTerminal.CONDITION_FALSE: ProspectiveDisposition.CONDITION_FALSE,
        NativeControllerUseTerminal.NONATTEMPT: ProspectiveDisposition.NONATTEMPT,
        NativeControllerUseTerminal.UNEVALUABLE: ProspectiveDisposition.UNEVALUABLE,
        NativeControllerUseTerminal.SOURCE_STOP: ProspectiveDisposition.UNEVALUABLE,
    }[value]


@dataclass(frozen=True, slots=True)
class HeldoutTargetComparison(CanonicalRecord):
    """One revealed target comparison with native-chain prerequisites."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/heldout-target-comparison'

    comparison_id: str
    prediction_level_bundle: ObjectIdentity
    prediction_bundle_commitment: ObjectIdentity
    bundle_level_count: int
    pre_reveal_bundle_issued: bool
    prediction_issue: StructuralPredictionIssue
    native_evidence: NativeHeldoutTargetEvidence
    observation: StructuralObservation
    match: PredictiveMatch
    restriction_profile_exact: bool
    native_admission_prerequisite_passed: bool
    native_prospective_validation_prerequisite_passed: bool
    comparison_evaluable: bool
    counterexample: bool
    positive_nonhold_prospective_validation: bool
    safe_abstention: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.comparison_id, field_name="comparison_id")
        issue = self.prediction_issue
        evidence = self.native_evidence
        observation = self.observation
        prediction = issue.primary_prediction
        if (
            self.prediction_level_bundle.object_schema != StructuralPredictionLevelBundle.SCHEMA
            or self.prediction_level_bundle.object_version
            != StructuralPredictionLevelBundle.VERSION
            or self.bundle_level_count != len(FOUR_TARGET_STRUCTURAL_RECURRENCE_LEVEL_BUNDLE_LEVELS)
            or not self.pre_reveal_bundle_issued
        ):
            raise ValueError('held-out comparison requires a complete pre-reveal measurement readiness--prospective use bundle')
        if (
            evidence.target_design != issue.target_design
            or evidence.achieved_level is not issue.comparison_level
            or evidence.evidence_world_id != prediction.evidence_world_id
            or observation.target_slot_id != prediction.target_slot_id
            or observation.target_level is not evidence.achieved_level
            or observation.evidence_world_id != evidence.evidence_world_id
            or observation.target_handoff != evidence.target_handoff
            or observation.evaluator != issue.evaluator
        ):
            raise ValueError("held-out comparison scope or frozen identities differ")
        if self.match.prediction != ObjectIdentity.from_record(
            prediction.prediction_id, prediction
        ) or self.match.observation != ObjectIdentity.from_record(
            observation.observation_id, observation
        ):
            raise ValueError("held-out match does not bind exact issued/revealed bytes")
        restriction_exact = prediction.restrictions == observation.restrictions
        level = _TARGET_LEVEL_INDEX[evidence.achieved_level]
        admission_required = level >= _TARGET_LEVEL_INDEX[TargetLevel.ADMISSION]
        prospective_validation_required = level >= _TARGET_LEVEL_INDEX[TargetLevel.PROSPECTIVE_USE]
        admission_passed = not admission_required or (
            evidence.admission_handoff is not None
            and evidence.native_admission_topology is observation.core.admission_topology
            and evidence.native_threshold_manifest == issue.target_design.native_threshold_manifest
        )
        prospective_validation_passed = not prospective_validation_required or (
            evidence.prospective_validation_issue is not None
            and evidence.prospective_validation_authority is not None
            and evidence.prospective_validation_handoff is not None
            and evidence.native_prospective_validation_terminal is not None
            and _structural_prospective_from_native(evidence.native_prospective_validation_terminal)
            is observation.core.prospective_validation_disposition
        )
        evaluable = admission_passed and prospective_validation_passed
        unsafe = bool(self.match.false_action_count or self.match.hidden_sink_admission_count)
        counterexample = evaluable and (
            not self.match.passed
            or not restriction_exact
            or unsafe
            or observation.ontology_counterexample
            or prediction.status is PredictionStatus.UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE
        )
        positive = bool(
            evaluable
            and not counterexample
            and evidence.achieved_level in {TargetLevel.PROSPECTIVE_USE, TargetLevel.CONTROLLER_USE_CEILING}
            and evidence.native_admission_topology
            in {AdmissionTopology.SINGLETON, AdmissionTopology.MULTIPLE}
            and evidence.native_prospective_validation_terminal is NativeControllerUseTerminal.VALIDATED
            and observation.core.policy_branch is PolicyBranch.EXACT_ACTION
            and observation.restrictions.action_role is ActionRole.REALIZATION_QUALIFIED
        )
        safe = bool(
            evaluable
            and not counterexample
            and self.match.passed
            and observation.core.policy_branch in {PolicyBranch.HOLD, PolicyBranch.NONATTEMPT}
            and evidence.native_prospective_validation_terminal
            not in {NativeControllerUseTerminal.VALIDATED, NativeControllerUseTerminal.OPPOSED}
        )
        reasons: set[str] = set()
        if not restriction_exact:
            reasons.add("RESTRICTION_PROFILE_MISMATCH")
        if not admission_passed:
            reasons.add("NATIVE_ADMISSION_PREREQUISITE_MISSING")
        if not prospective_validation_passed:
            reasons.add("NATIVE_CONTROLLER_USE_PREREQUISITE_MISSING")
        if not self.match.passed:
            reasons.add("ISSUED_PREDICTION_MISMATCH")
        if unsafe:
            reasons.add("SAFETY_CRITICAL_MISMATCH")
        if observation.ontology_counterexample or prediction.status is (
            PredictionStatus.UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE
        ):
            reasons.add("UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE")
        expected_reasons = tuple(sorted(reasons))
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.restriction_profile_exact != restriction_exact
            or self.native_admission_prerequisite_passed != admission_passed
            or self.native_prospective_validation_prerequisite_passed != prospective_validation_passed
            or self.comparison_evaluable != evaluable
            or self.counterexample != counterexample
            or self.positive_nonhold_prospective_validation != positive
            or self.safe_abstention != safe
            or self.reason_codes != expected_reasons
        ):
            raise ValueError("held-out comparison flags are not mechanically derived")


def compare_heldout_target(
    *,
    comparison_id: str,
    prediction_bundle: StructuralPredictionLevelBundle,
    native_evidence: NativeHeldoutTargetEvidence,
    observation: StructuralObservation,
) -> HeldoutTargetComparison:
    """Match one immutable issue only after target-owned native adjudication."""

    prediction_issue = select_issued_prediction_for_achieved_level(
        prediction_bundle,
        achieved_level=native_evidence.achieved_level,
    )
    match = PredictiveStructuralMatcher().match(
        prediction_issue.primary_prediction,
        observation,
    )
    restriction_exact = prediction_issue.primary_prediction.restrictions == observation.restrictions
    level = _TARGET_LEVEL_INDEX[native_evidence.achieved_level]
    admission_required = level >= _TARGET_LEVEL_INDEX[TargetLevel.ADMISSION]
    prospective_validation_required = level >= _TARGET_LEVEL_INDEX[TargetLevel.PROSPECTIVE_USE]
    admission_passed = not admission_required or (
        native_evidence.admission_handoff is not None
        and native_evidence.native_admission_topology is observation.core.admission_topology
        and native_evidence.native_threshold_manifest
        == prediction_issue.target_design.native_threshold_manifest
    )
    prospective_validation_passed = not prospective_validation_required or (
        native_evidence.prospective_validation_issue is not None
        and native_evidence.prospective_validation_authority is not None
        and native_evidence.prospective_validation_handoff is not None
        and native_evidence.native_prospective_validation_terminal is not None
        and _structural_prospective_from_native(native_evidence.native_prospective_validation_terminal)
        is observation.core.prospective_validation_disposition
    )
    evaluable = admission_passed and prospective_validation_passed
    unsafe = bool(match.false_action_count or match.hidden_sink_admission_count)
    counterexample = evaluable and (
        not match.passed
        or not restriction_exact
        or unsafe
        or observation.ontology_counterexample
        or prediction_issue.primary_prediction.status
        is PredictionStatus.UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE
    )
    positive = bool(
        evaluable
        and not counterexample
        and native_evidence.achieved_level in {TargetLevel.PROSPECTIVE_USE, TargetLevel.CONTROLLER_USE_CEILING}
        and native_evidence.native_admission_topology
        in {AdmissionTopology.SINGLETON, AdmissionTopology.MULTIPLE}
        and native_evidence.native_prospective_validation_terminal is NativeControllerUseTerminal.VALIDATED
        and observation.core.policy_branch is PolicyBranch.EXACT_ACTION
        and observation.restrictions.action_role is ActionRole.REALIZATION_QUALIFIED
    )
    safe = bool(
        evaluable
        and not counterexample
        and match.passed
        and observation.core.policy_branch in {PolicyBranch.HOLD, PolicyBranch.NONATTEMPT}
        and native_evidence.native_prospective_validation_terminal
        not in {NativeControllerUseTerminal.VALIDATED, NativeControllerUseTerminal.OPPOSED}
    )
    reasons: set[str] = set()
    if not restriction_exact:
        reasons.add("RESTRICTION_PROFILE_MISMATCH")
    if not admission_passed:
        reasons.add("NATIVE_ADMISSION_PREREQUISITE_MISSING")
    if not prospective_validation_passed:
        reasons.add("NATIVE_CONTROLLER_USE_PREREQUISITE_MISSING")
    if not match.passed:
        reasons.add("ISSUED_PREDICTION_MISMATCH")
    if unsafe:
        reasons.add("SAFETY_CRITICAL_MISMATCH")
    if observation.ontology_counterexample or prediction_issue.primary_prediction.status is (
        PredictionStatus.UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE
    ):
        reasons.add("UNIVERSAL_ONTOLOGY_COUNTEREXAMPLE")
    return HeldoutTargetComparison(
        comparison_id=comparison_id,
        prediction_level_bundle=ObjectIdentity.from_record(
            prediction_bundle.bundle_id,
            prediction_bundle,
        ),
        prediction_bundle_commitment=prediction_bundle.bundle_commitment,
        bundle_level_count=len(prediction_bundle.level_issues),
        pre_reveal_bundle_issued=prediction_bundle.issued_before_target_outcomes,
        prediction_issue=prediction_issue,
        native_evidence=native_evidence,
        observation=observation,
        match=match,
        restriction_profile_exact=restriction_exact,
        native_admission_prerequisite_passed=admission_passed,
        native_prospective_validation_prerequisite_passed=prospective_validation_passed,
        comparison_evaluable=evaluable,
        counterexample=counterexample,
        positive_nonhold_prospective_validation=positive,
        safe_abstention=safe,
        reason_codes=tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class HeldoutStructuralAdjudication(CanonicalRecord):
    'Held-out local structural meta-adjudication with its local-law evidence ceiling.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/heldout-structural-adjudication'

    adjudication_id: str
    compiler_freeze: StructuralCompilerFreeze
    comparisons: tuple[HeldoutTargetComparison, ...]
    missing_target_slot_ids: tuple[str, ...]
    counterexample_target_slot_ids: tuple[str, ...]
    unevaluable_target_slot_ids: tuple[str, ...]
    positive_quantum_target_slot_ids: tuple[str, ...]
    positive_cross_substrate_target_slot_ids: tuple[str, ...]
    safe_abstention_target_slot_ids: tuple[str, ...]
    independent_positive_member_count: int
    verdict: HeldoutStructuralVerdict
    reason_codes: tuple[str, ...]
    universality_proof_claimed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        require_sorted_unique_ids(
            self.comparisons,
            attribute="comparison_id",
            field_name="comparisons",
        )
        slot_ids = tuple(
            comparison.native_evidence.target_design.target_slot.target_slot_id
            for comparison in self.comparisons
        )
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("terminal adjudication has duplicate target comparisons")
        frozen_slots = {
            design.target_slot.target_slot_id for design in self.compiler_freeze.target_designs
        }
        if not set(slot_ids) <= frozen_slots:
            raise ValueError("terminal comparison contains an unfrozen target")
        freeze_identity = ObjectIdentity.from_record(
            self.compiler_freeze.freeze_id,
            self.compiler_freeze,
        )
        if any(
            ObjectIdentity.from_record(
                comparison.prediction_issue.compiler_freeze.freeze_id,
                comparison.prediction_issue.compiler_freeze,
            )
            != freeze_identity
            for comparison in self.comparisons
        ):
            raise ValueError("terminal comparisons use different compiler freezes")
        missing = tuple(sorted(frozen_slots - set(slot_ids)))
        counterexamples = tuple(
            sorted(
                comparison.native_evidence.target_design.target_slot.target_slot_id
                for comparison in self.comparisons
                if comparison.counterexample
            )
        )
        unevaluable = tuple(
            sorted(
                comparison.native_evidence.target_design.target_slot.target_slot_id
                for comparison in self.comparisons
                if not comparison.comparison_evaluable
            )
        )
        quantum = tuple(
            sorted(
                comparison.native_evidence.target_design.target_slot.target_slot_id
                for comparison in self.comparisons
                if comparison.positive_nonhold_prospective_validation
                and comparison.native_evidence.target_design.target_kind
                is HeldoutTargetKind.HELDOUT_QUANTUM
            )
        )
        cross = tuple(
            sorted(
                comparison.native_evidence.target_design.target_slot.target_slot_id
                for comparison in self.comparisons
                if comparison.positive_nonhold_prospective_validation
                and comparison.native_evidence.target_design.target_kind
                is HeldoutTargetKind.HELDOUT_CROSS_SUBSTRATE
            )
        )
        abstentions = tuple(
            sorted(
                comparison.native_evidence.target_design.target_slot.target_slot_id
                for comparison in self.comparisons
                if comparison.safe_abstention
            )
        )
        member_count = len(quantum) + len(cross)
        if counterexamples:
            verdict = HeldoutStructuralVerdict.STRUCTURAL_COMPILER_COUNTEREXAMPLE
        elif missing or unevaluable or not self.comparisons:
            verdict = HeldoutStructuralVerdict.HELDOUT_TARGETS_UNEVALUABLE
        elif quantum and cross:
            verdict = HeldoutStructuralVerdict.HELDOUT_ADMISSION_CONTROL_RECURRENCE_SUPPORTED
        elif len(abstentions) == len(self.comparisons):
            verdict = HeldoutStructuralVerdict.HELDOUT_SAFE_ABSTENTION_CALIBRATED
        else:
            verdict = HeldoutStructuralVerdict.HELDOUT_PROCEDURAL_RECURRENCE_ONLY
        reasons: set[str] = set()
        if counterexamples:
            reasons.add("STRUCTURAL_PREDICTION_EXCLUDED_VALID_OUTCOME")
        if missing:
            reasons.add("FROZEN_TARGET_RESULT_MISSING")
        if unevaluable:
            reasons.add("NATIVE_COMPARISON_PREREQUISITE_MISSING")
        if verdict is HeldoutStructuralVerdict.HELDOUT_PROCEDURAL_RECURRENCE_ONLY:
            reasons.add('POSITIVE_QUANTUM_AND_CROSS_SUBSTRATE_PROSPECTIVE_VALIDATION_NOT_BOTH_PRESENT')
        if verdict is HeldoutStructuralVerdict.HELDOUT_SAFE_ABSTENTION_CALIBRATED:
            reasons.add('NO_POSITIVE_PROSPECTIVE_VALIDATION_MEMBER')
        for field_name, expected in (
            ("missing_target_slot_ids", missing),
            ("counterexample_target_slot_ids", counterexamples),
            ("unevaluable_target_slot_ids", unevaluable),
            ("positive_quantum_target_slot_ids", quantum),
            ("positive_cross_substrate_target_slot_ids", cross),
            ("safe_abstention_target_slot_ids", abstentions),
        ):
            values = getattr(self, field_name)
            require_sorted_unique_strings(values, field_name=field_name)
            if values != expected:
                raise ValueError(f"{field_name} is not derived from target comparisons")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.independent_positive_member_count != member_count
            or self.verdict is not verdict
            or self.reason_codes != tuple(sorted(reasons))
            or self.universality_proof_claimed
        ):
            raise ValueError("terminal structural verdict fields are not mechanically derived")


def adjudicate_heldout_targets(
    *,
    adjudication_id: str,
    compiler_freeze: StructuralCompilerFreeze,
    comparisons: tuple[HeldoutTargetComparison, ...],
) -> HeldoutStructuralAdjudication:
    """Select exactly one frozen terminal meta-class without pooled statistics."""

    ordered = tuple(sorted(comparisons, key=lambda value: value.comparison_id))
    frozen_slots = {design.target_slot.target_slot_id for design in compiler_freeze.target_designs}
    entered_slots = {
        comparison.native_evidence.target_design.target_slot.target_slot_id
        for comparison in ordered
    }
    missing = tuple(sorted(frozen_slots - entered_slots))
    counterexamples = tuple(
        sorted(
            comparison.native_evidence.target_design.target_slot.target_slot_id
            for comparison in ordered
            if comparison.counterexample
        )
    )
    unevaluable = tuple(
        sorted(
            comparison.native_evidence.target_design.target_slot.target_slot_id
            for comparison in ordered
            if not comparison.comparison_evaluable
        )
    )
    quantum = tuple(
        sorted(
            comparison.native_evidence.target_design.target_slot.target_slot_id
            for comparison in ordered
            if comparison.positive_nonhold_prospective_validation
            and comparison.native_evidence.target_design.target_kind
            is HeldoutTargetKind.HELDOUT_QUANTUM
        )
    )
    cross = tuple(
        sorted(
            comparison.native_evidence.target_design.target_slot.target_slot_id
            for comparison in ordered
            if comparison.positive_nonhold_prospective_validation
            and comparison.native_evidence.target_design.target_kind
            is HeldoutTargetKind.HELDOUT_CROSS_SUBSTRATE
        )
    )
    abstentions = tuple(
        sorted(
            comparison.native_evidence.target_design.target_slot.target_slot_id
            for comparison in ordered
            if comparison.safe_abstention
        )
    )
    if counterexamples:
        verdict = HeldoutStructuralVerdict.STRUCTURAL_COMPILER_COUNTEREXAMPLE
    elif missing or unevaluable or not ordered:
        verdict = HeldoutStructuralVerdict.HELDOUT_TARGETS_UNEVALUABLE
    elif quantum and cross:
        verdict = HeldoutStructuralVerdict.HELDOUT_ADMISSION_CONTROL_RECURRENCE_SUPPORTED
    elif len(abstentions) == len(ordered):
        verdict = HeldoutStructuralVerdict.HELDOUT_SAFE_ABSTENTION_CALIBRATED
    else:
        verdict = HeldoutStructuralVerdict.HELDOUT_PROCEDURAL_RECURRENCE_ONLY
    reasons: set[str] = set()
    if counterexamples:
        reasons.add("STRUCTURAL_PREDICTION_EXCLUDED_VALID_OUTCOME")
    if missing:
        reasons.add("FROZEN_TARGET_RESULT_MISSING")
    if unevaluable:
        reasons.add("NATIVE_COMPARISON_PREREQUISITE_MISSING")
    if verdict is HeldoutStructuralVerdict.HELDOUT_PROCEDURAL_RECURRENCE_ONLY:
        reasons.add('POSITIVE_QUANTUM_AND_CROSS_SUBSTRATE_PROSPECTIVE_VALIDATION_NOT_BOTH_PRESENT')
    if verdict is HeldoutStructuralVerdict.HELDOUT_SAFE_ABSTENTION_CALIBRATED:
        reasons.add('NO_POSITIVE_PROSPECTIVE_VALIDATION_MEMBER')
    return HeldoutStructuralAdjudication(
        adjudication_id=adjudication_id,
        compiler_freeze=compiler_freeze,
        comparisons=ordered,
        missing_target_slot_ids=missing,
        counterexample_target_slot_ids=counterexamples,
        unevaluable_target_slot_ids=unevaluable,
        positive_quantum_target_slot_ids=quantum,
        positive_cross_substrate_target_slot_ids=cross,
        safe_abstention_target_slot_ids=abstentions,
        independent_positive_member_count=len(quantum) + len(cross),
        verdict=verdict,
        reason_codes=tuple(sorted(reasons)),
        universality_proof_claimed=False,
    )


def decode_structural_prediction_input(payload: bytes) -> StructuralPredictionInput:
    return decode_canonical_bytes(
        payload,
        StructuralPredictionInput,
        maximum_bytes=MAX_STRUCTURAL_RECURRENCE_RECORD_BYTES,
    )


def decode_structural_prediction(payload: bytes) -> StructuralPrediction:
    return decode_canonical_bytes(
        payload,
        StructuralPrediction,
        maximum_bytes=MAX_STRUCTURAL_RECURRENCE_RECORD_BYTES,
    )


def decode_structural_observation(payload: bytes) -> StructuralObservation:
    return decode_canonical_bytes(
        payload,
        StructuralObservation,
        maximum_bytes=MAX_STRUCTURAL_RECURRENCE_RECORD_BYTES,
    )


def decode_predictive_match(payload: bytes) -> PredictiveMatch:
    return decode_canonical_bytes(
        payload,
        PredictiveMatch,
        maximum_bytes=MAX_STRUCTURAL_RECURRENCE_RECORD_BYTES,
    )


def decode_structural_donor_corpus(payload: bytes) -> StructuralDonorCorpus:
    return decode_canonical_bytes(
        payload,
        StructuralDonorCorpus,
        maximum_bytes=MAX_STRUCTURAL_RECURRENCE_RECORD_BYTES,
    )


def decode_truth_known_conformance_result(payload: bytes) -> StructuralRecurrenceTruthKnownConformanceResult:
    return decode_canonical_bytes(
        payload,
        StructuralRecurrenceTruthKnownConformanceResult,
        maximum_bytes=MAX_STRUCTURAL_RECURRENCE_RECORD_BYTES,
    )


def decode_structural_compiler_freeze(payload: bytes) -> StructuralCompilerFreeze:
    return decode_canonical_bytes(
        payload,
        StructuralCompilerFreeze,
        maximum_bytes=MAX_STRUCTURAL_RECURRENCE_RECORD_BYTES,
    )


def decode_structural_prediction_issue(payload: bytes) -> StructuralPredictionIssue:
    return decode_canonical_bytes(
        payload,
        StructuralPredictionIssue,
        maximum_bytes=MAX_STRUCTURAL_RECURRENCE_RECORD_BYTES,
    )


def decode_structural_prediction_level_bundle(
    payload: bytes,
) -> StructuralPredictionLevelBundle:
    return decode_canonical_bytes(
        payload,
        StructuralPredictionLevelBundle,
        maximum_bytes=MAX_STRUCTURAL_RECURRENCE_RECORD_BYTES,
    )


def decode_heldout_target_design_spec(payload: bytes) -> HeldoutTargetDesignSpec:
    return decode_canonical_bytes(
        payload,
        HeldoutTargetDesignSpec,
        maximum_bytes=MAX_STRUCTURAL_RECURRENCE_RECORD_BYTES,
    )


def decode_native_heldout_target_evidence(
    payload: bytes,
) -> NativeHeldoutTargetEvidence:
    return decode_canonical_bytes(
        payload,
        NativeHeldoutTargetEvidence,
        maximum_bytes=MAX_STRUCTURAL_RECURRENCE_RECORD_BYTES,
    )


def decode_heldout_target_comparison(payload: bytes) -> HeldoutTargetComparison:
    return decode_canonical_bytes(
        payload,
        HeldoutTargetComparison,
        maximum_bytes=MAX_STRUCTURAL_RECURRENCE_RECORD_BYTES,
    )


def decode_heldout_structural_adjudication(
    payload: bytes,
) -> HeldoutStructuralAdjudication:
    return decode_canonical_bytes(
        payload,
        HeldoutStructuralAdjudication,
        maximum_bytes=MAX_STRUCTURAL_RECURRENCE_RECORD_BYTES,
    )


__all__ = [
    "ADMISSION_ROLES",
    "COMPILER_INPUT_SCHEMA_IDS",
    "COMPILER_OUTPUT_SCHEMA_IDS",
    "CORE_FIELD_IDS",
    'MAX_STRUCTURAL_RECURRENCE_RECORD_BYTES',
    "FOUR_TARGET_STRUCTURAL_RECURRENCE_FORBIDDEN_CLAIM_IDS",
    "FOUR_TARGET_STRUCTURAL_RECURRENCE_LEVEL_BUNDLE_LEVELS",
    "FOUR_TARGET_STRUCTURAL_RECURRENCE_TARGET_SLOT_IDS",
    "REQUIRED_TRUTH_KNOWN_FIXTURE_KINDS",
    "REQUIRED_DONOR_FAMILIES",
    "RESTRICTION_FIELD_IDS",
    "UNIVERSAL_ROLES",
    "ActionCandidateFact",
    "ActionRole",
    "AdmissionOperandFact",
    "AdmissionTopology",
    "BoundaryRole",
    'StructuralRecurrenceTruthKnownConformanceResult',
    'StructuralRecurrenceTruthKnownFixtureKind',
    'StructuralRecurrenceTruthKnownFixtureResult',
    "DenominatorStructure",
    "DevelopmentControllerUseExpectation",
    "DonorEvidenceRecord",
    "DonorFamily",
    "DonorCorpusIdentity",
    "HeldoutStructuralAdjudication",
    "HeldoutStructuralVerdict",
    "HeldoutTargetComparison",
    "HeldoutTargetDesign",
    "HeldoutTargetDesignSpec",
    "HeldoutTargetKind",
    "HistoryClockQuotient",
    "HistoryRole",
    "InternalTransferRole",
    "MethodQualificationVerdict",
    "NativeHeldoutTargetEvidence",
    "NativeControllerUseTerminal",
    "NativeRoleBinding",
    "NativeRoleCompatibilityRecord",
    "NumericalRole",
    "ObserverStatus",
    "OperandStatus",
    'ProspectiveDisposition',
    "PolicyBranch",
    "PredictionStatus",
    "PredictiveComparatorPanel",
    "PredictiveFieldMatch",
    "PredictiveMatch",
    "PredictiveRecurrenceAdjudication",
    "PredictiveStructuralAdjudicator",
    "PredictiveStructuralMatcher",
    "PredictiveStructuralRecurrenceCompiler",
    "PredictorKind",
    "PreparationRole",
    "ReceiverRole",
    "StructuralCorePrediction",
    "StructuralCompilerFreeze",
    "StructuralDonorCorpus",
    "StructuralObservation",
    "StructuralObservedCore",
    "StructuralOntologyIdentity",
    "StructuralPrediction",
    "StructuralPredictionIssue",
    "StructuralPredictionLevelBundle",
    "StructuralPredictionInput",
    "StructuralReason",
    "StructuralRestrictionProfile",
    "StructuralSafetyObservation",
    "StructuralSafetyPrediction",
    "SupportTransport",
    "TargetLocalityAudit",
    "TargetLevel",
    "TargetSlotRecord",
    "TopologyModeRole",
    "UniversalRole",
    "always_act_comparator",
    "always_hold_comparator",
    "adjudicate_heldout_targets",
    "build_truth_known_conformance",
    "build_structural_recurrence_donor_corpus",
    "build_structural_recurrence_heldout_target_designs",
    "build_structural_recurrence_compiler_freeze",
    "compare_heldout_target",
    "decode_truth_known_conformance_result",
    "decode_heldout_structural_adjudication",
    "decode_heldout_target_design_spec",
    "decode_heldout_target_comparison",
    "decode_native_heldout_target_evidence",
    "decode_predictive_match",
    "decode_structural_compiler_freeze",
    "decode_structural_donor_corpus",
    "decode_structural_observation",
    "decode_structural_prediction",
    "decode_structural_prediction_issue",
    "decode_structural_prediction_level_bundle",
    "decode_structural_prediction_input",
    "evaluate_truth_known_fixture",
    "history_free_comparator",
    "issue_structural_prediction",
    "issue_structural_prediction_level_bundle",
    "make_donor_evidence_record",
    "make_prediction_donor_identity",
    "qualify_truth_known_conformance",
    "response_only_comparator",
    "select_issued_prediction_for_achieved_level",
    'structural_ontology',
    "wildcard_comparator",
]
