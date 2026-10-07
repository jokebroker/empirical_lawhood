"""Outcome-blind governance and phase-gate records for physical scale morphism.

The physical morphism challenge cannot manufacture independent specification,
laboratory evidence, authority, or outcome access inside the repository.  This
module supplies the strict package-local records that let those externally
owned facts enter the existing candidate/runtime machinery without being
collapsed into booleans or prose.  It performs no I/O and grants no authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)

from .contracts import PhysicalScaleMorphismEvidenceWorld, PhysicalScaleMorphismLossiness, PhysicalScaleMorphismObstructionSet
from .inference import PhysicalScaleMorphismPowerQualification, PhysicalScaleMorphismPowerTerminal


PHYSICAL_SCALE_MORPHISM_NATIVE_DOSSIER_SECTION_IDS = (
    "apparatus-identity",
    "calibration-and-failure-semantics",
    "circuit-topology-and-components",
    "decisive-apparatus-invalidators",
    "fabrication-tolerances-and-dimensions",
    "native-actions-clocks-and-units",
    "native-preparations-and-measurements",
    "operating-and-safety-regions",
    "physical-scale-change",
    "reset-and-equilibration",
    "target-and-hold-semantics",
)

PHYSICAL_SCALE_MORPHISM_CONSTRUCT_REVIEW_QUESTION_IDS = (
    "action-stages-observable",
    "hidden-mode-preparations-realizable",
    "lab-executable-without-expected-labels",
    "map-direction-and-lossiness-correct",
    "physical-scale-genuinely-changed",
    "quantities-dimensionless-or-native-local",
    "receiver-and-numerical-maps-distinct",
    "safety-failure-distinct-from-zero-response",
)

PHYSICAL_SCALE_MORPHISM_REQUIRED_PARTY_ROLES = (
    "adjudicator",
    "construct-reviewer",
    "custodian",
    "evaluator",
    "lab-team",
    "map-team",
    "method-team",
    "native-owner",
)

PHYSICAL_SCALE_MORPHISM_APPARATUS_GATE_IDS = (
    "acquisition",
    "batch-identity",
    "causal-chronology",
    "delivery",
    "loading",
    "measured-hold",
    "metrology",
    "preparation",
    "repeatability",
    "reset",
    "safety",
    "storage-custody",
    "topology",
)

PHYSICAL_SCALE_MORPHISM_PHYSICAL_SCALE_CELLS = (16, 32, 64)

PHYSICAL_SCALE_MORPHISM_PREDICTION_FREEZE_REQUIRED_OBJECT_SCHEMAS = (
    'empirical-lawhood/simulators/rc-ladder-response/rc-ladder-response-numerical-qualification',
    'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-comparator-encoding',
    'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-defect-thresholds',
    'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-forecast-triplet',
    'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-history-candidate',
    'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-morphism-composition-contract',
    'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-morphism-record',
    'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-normalization-spec',
    'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-normalization-uncertainty',
    'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-receiver-spec',
)


class PhysicalScaleMorphismPartyRole(StrEnum):
    ADJUDICATOR = "adjudicator"
    CONSTRUCT_REVIEWER = "construct-reviewer"
    CUSTODIAN = "custodian"
    EVALUATOR = "evaluator"
    LAB_TEAM = "lab-team"
    MAP_TEAM = "map-team"
    METHOD_TEAM = "method-team"
    NATIVE_OWNER = "native-owner"


class PhysicalScaleMorphismReviewAnswer(StrEnum):
    NO = "NO"
    UNEVALUABLE = "UNEVALUABLE"
    YES = "YES"


class PhysicalScaleMorphismAccessClass(StrEnum):
    DEVELOPMENT_RESPONSE = "DEVELOPMENT_RESPONSE"
    DOSSIER = "DOSSIER"
    EVALUATION_OUTCOME = "EVALUATION_OUTCOME"
    METHOD_SOURCE = "METHOD_SOURCE"
    PREDICTION = "PREDICTION"


class PhysicalScaleMorphismConstructTerminal(StrEnum):
    CONSTRUCT_INDEPENDENCE_NOT_QUALIFIED = "CONSTRUCT_INDEPENDENCE_NOT_QUALIFIED"
    CONSTRUCT_INDEPENDENCE_QUALIFIED = "CONSTRUCT_INDEPENDENCE_QUALIFIED"


class PhysicalScaleMorphismGateState(StrEnum):
    FAIL = "FAIL"
    PASS = "PASS"
    UNEVALUABLE = "UNEVALUABLE"


class PhysicalScaleMorphismSourceApparatusTerminal(StrEnum):
    SOURCE_AND_APPARATUS_QUALIFIED = "SOURCE_AND_APPARATUS_QUALIFIED"
    SOURCE_OR_APPARATUS_NOT_QUALIFIED = "SOURCE_OR_APPARATUS_NOT_QUALIFIED"


class PhysicalScaleMorphismImplementationRecurrenceState(StrEnum):
    OPPOSED = "OPPOSED"
    SUPPORTED = "SUPPORTED"
    NOT_TESTED = "NOT_TESTED"
    UNEVALUABLE = "UNEVALUABLE"


class PhysicalScaleMorphismSimulationState(StrEnum):
    OPPOSED = "OPPOSED"
    SUPPORTED = "SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"


class PhysicalScaleMorphismIssueTerminal(StrEnum):
    ISSUED_COMPLETE_ROSTER = "ISSUED_COMPLETE_ROSTER"
    ISSUE_OR_BOARD_QUALIFICATION_STOP = "ISSUE_OR_BOARD_QUALIFICATION_STOP"


def _require_identity(value: ObjectIdentity, schema: str, *, field_name: str) -> None:
    if value.object_schema != schema:
        raise ValueError(f"{field_name} has the wrong object schema")


def _require_sorted_roles(values: tuple[PhysicalScaleMorphismPartyRole, ...], *, field_name: str) -> None:
    if tuple(sorted(set(values), key=lambda value: value.value)) != values:
        raise ValueError(f"{field_name} must be sorted and unique")


@dataclass(frozen=True, slots=True)
class IndependentApparatusDossier(CanonicalRecord):
    """Hash-locked native apparatus/task dossier metadata, never its payload."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/independent-apparatus-dossier'

    dossier_id: str
    native_owner_id: str
    native_author_ids: tuple[str, ...]
    payload_sha256: str
    payload_size_bytes: int
    external_locator_id: str
    section_ids: tuple[str, ...]
    native_object_ids: tuple[str, ...]
    decisive_invalidator_ids: tuple[str, ...]
    lock_event_id: str
    apparatus_native_language_attested: bool
    prediction_artifact_ids_visible_before_lock: tuple[str, ...]
    repository_prediction_label_count_at_lock: int
    protected_outcome_count_at_lock: int
    outcome_access: OutcomeAccess
    locked: bool

    def __post_init__(self) -> None:
        for name in ("dossier_id", "native_owner_id", "external_locator_id", "lock_event_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.native_author_ids, field_name="native_author_ids", allow_empty=False
        )
        if self.native_owner_id not in self.native_author_ids:
            raise ValueError("native owner must be in the dossier author roster")
        validate_sha256(self.payload_sha256, field_name="payload_sha256")
        if self.payload_size_bytes <= 0:
            raise ValueError("native dossier payload size must be positive")
        if self.section_ids != PHYSICAL_SCALE_MORPHISM_NATIVE_DOSSIER_SECTION_IDS:
            raise ValueError("native dossier does not cover the frozen section roster")
        require_sorted_unique_strings(
            self.native_object_ids, field_name="native_object_ids", allow_empty=False
        )
        require_sorted_unique_strings(
            self.decisive_invalidator_ids,
            field_name="decisive_invalidator_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.prediction_artifact_ids_visible_before_lock,
            field_name="prediction_artifact_ids_visible_before_lock",
        )
        if not self.apparatus_native_language_attested:
            raise ValueError("native dossier lacks its apparatus-language attestation")
        if (
            self.prediction_artifact_ids_visible_before_lock
            or self.repository_prediction_label_count_at_lock
            or self.protected_outcome_count_at_lock
        ):
            raise ValueError("native dossier crossed its independent pre-lock information boundary")
        if (
            self.repository_prediction_label_count_at_lock < 0
            or self.protected_outcome_count_at_lock < 0
        ):
            raise ValueError("dossier access counts must be nonnegative")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("native dossier must be locked outcome blind")
        if not self.locked:
            raise ValueError("native dossier record must describe a completed lock")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismConstructMapEntry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-construct-map-entry'

    entry_id: str
    native_object_id: str
    repository_object_schema: str
    compatibility_map_id: str
    direction: str
    lossiness: PhysicalScaleMorphismLossiness
    unit_transform_id: str
    native_semantics_changed: bool

    def __post_init__(self) -> None:
        for name in ("entry_id", "native_object_id", "compatibility_map_id", "unit_transform_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_schema(self.repository_object_schema)
        validate_nonempty(self.direction, field_name="direction")
        if self.native_semantics_changed:
            raise ValueError("construct mapping cannot change native semantics")


@dataclass(frozen=True, slots=True)
class IndependentConstructMapping(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/independent-construct-mapping'

    mapping_id: str
    dossier: ObjectIdentity
    map_team_ids: tuple[str, ...]
    entries: tuple[PhysicalScaleMorphismConstructMapEntry, ...]
    predicted_survivor_table_accessed: bool
    evaluation_response_artifact_ids_accessed: tuple[str, ...]
    outcome_access: OutcomeAccess
    frozen: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.mapping_id, field_name="mapping_id")
        _require_identity(self.dossier, IndependentApparatusDossier.SCHEMA, field_name="dossier")
        require_sorted_unique_strings(
            self.map_team_ids, field_name="map_team_ids", allow_empty=False
        )
        require_sorted_unique_ids(self.entries, attribute="entry_id", field_name="entries")
        if not self.entries:
            raise ValueError("construct mapping cannot be empty")
        native_ids = tuple(value.native_object_id for value in self.entries)
        if tuple(sorted(set(native_ids))) != native_ids:
            raise ValueError("construct mapping must map each native object exactly once")
        require_sorted_unique_strings(
            self.evaluation_response_artifact_ids_accessed,
            field_name="evaluation_response_artifact_ids_accessed",
        )
        if self.predicted_survivor_table_accessed or self.evaluation_response_artifact_ids_accessed:
            raise ValueError("construct mapping crossed its outcome/prediction firewall")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("construct mapping must be authored outcome blind")
        if not self.frozen:
            raise ValueError("construct mapping must be frozen before review")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismConstructReviewItem(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-construct-review-item'

    question_id: str
    answer: PhysicalScaleMorphismReviewAnswer
    material_issue_ids: tuple[str, ...]
    resolution_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.question_id, field_name="question_id")
        if self.question_id not in PHYSICAL_SCALE_MORPHISM_CONSTRUCT_REVIEW_QUESTION_IDS:
            raise ValueError("construct review contains an unknown question")
        require_sorted_unique_strings(self.material_issue_ids, field_name="material_issue_ids")
        require_sorted_unique_strings(self.resolution_ids, field_name="resolution_ids")
        if self.answer is PhysicalScaleMorphismReviewAnswer.YES and self.material_issue_ids:
            raise ValueError("affirmative construct answer cannot retain a material issue")
        if self.answer is not PhysicalScaleMorphismReviewAnswer.YES and not self.material_issue_ids:
            raise ValueError("non-affirmative construct answer requires a material issue")


@dataclass(frozen=True, slots=True)
class ConstructIndependenceReview(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/construct-independence-review'

    review_id: str
    dossier: ObjectIdentity
    mapping: ObjectIdentity
    reviewer_id: str
    native_owner_ids: tuple[str, ...]
    map_team_ids: tuple[str, ...]
    items: tuple[PhysicalScaleMorphismConstructReviewItem, ...]
    reviewer_conflict_ids: tuple[str, ...]
    dossier_prediction_blinding_confirmed: bool
    laboratory_label_blinding_executable: bool
    role_separation_confirmed: bool
    protected_outcome_count_at_review: int
    outcome_access: OutcomeAccess
    qualified: bool
    frozen: bool

    def __post_init__(self) -> None:
        for name in ("review_id", "reviewer_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        _require_identity(self.dossier, IndependentApparatusDossier.SCHEMA, field_name="dossier")
        _require_identity(self.mapping, IndependentConstructMapping.SCHEMA, field_name="mapping")
        require_sorted_unique_strings(
            self.native_owner_ids, field_name="native_owner_ids", allow_empty=False
        )
        require_sorted_unique_strings(
            self.map_team_ids, field_name="map_team_ids", allow_empty=False
        )
        if self.reviewer_id in set(self.native_owner_ids) | set(self.map_team_ids):
            raise ValueError("construct reviewer must be independent of both authorships")
        require_sorted_unique_ids(self.items, attribute="question_id", field_name="items")
        if tuple(value.question_id for value in self.items) != PHYSICAL_SCALE_MORPHISM_CONSTRUCT_REVIEW_QUESTION_IDS:
            raise ValueError("construct review does not cover all frozen questions")
        require_sorted_unique_strings(
            self.reviewer_conflict_ids, field_name="reviewer_conflict_ids"
        )
        if self.protected_outcome_count_at_review < 0:
            raise ValueError("review outcome count must be nonnegative")
        expected = all(value.answer is PhysicalScaleMorphismReviewAnswer.YES for value in self.items) and all(
            (
                not self.reviewer_conflict_ids,
                self.dossier_prediction_blinding_confirmed,
                self.laboratory_label_blinding_executable,
                self.role_separation_confirmed,
                self.protected_outcome_count_at_review == 0,
                self.outcome_access is OutcomeAccess.OUTCOME_BLIND,
                self.frozen,
            )
        )
        if self.qualified != expected:
            raise ValueError("construct qualification is not derived from the review")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismRoleAssignment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-role-assignment'

    assignment_id: str
    actor_id: str
    roles: tuple[PhysicalScaleMorphismPartyRole, ...]
    overlap_declared: bool

    def __post_init__(self) -> None:
        for name in ("assignment_id", "actor_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        _require_sorted_roles(self.roles, field_name="roles")
        if not self.roles:
            raise ValueError("role assignment cannot be empty")
        if self.overlap_declared != (len(self.roles) > 1):
            raise ValueError("role overlap declaration is not derived from the roster")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismAccessEvent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-access-event'

    event_id: str
    event_ordinal: int
    actor_id: str
    artifact_id: str
    access_class: PhysicalScaleMorphismAccessClass
    outcome_access: OutcomeAccess
    before_dossier_lock: bool

    def __post_init__(self) -> None:
        for name in ("event_id", "actor_id", "artifact_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.event_ordinal < 0:
            raise ValueError("access-event ordinal must be nonnegative")


def _access_separation_qualified(
    assignments: tuple[PhysicalScaleMorphismRoleAssignment, ...], events: tuple[PhysicalScaleMorphismAccessEvent, ...]
) -> bool:
    role_by_actor = {assignment.actor_id: set(assignment.roles) for assignment in assignments}
    if len(role_by_actor) != len(assignments):
        return False
    sensitive_pairs = (
        {PhysicalScaleMorphismPartyRole.NATIVE_OWNER, PhysicalScaleMorphismPartyRole.METHOD_TEAM},
        {PhysicalScaleMorphismPartyRole.NATIVE_OWNER, PhysicalScaleMorphismPartyRole.EVALUATOR},
        {PhysicalScaleMorphismPartyRole.METHOD_TEAM, PhysicalScaleMorphismPartyRole.EVALUATOR},
        {PhysicalScaleMorphismPartyRole.CONSTRUCT_REVIEWER, PhysicalScaleMorphismPartyRole.MAP_TEAM},
        {PhysicalScaleMorphismPartyRole.CONSTRUCT_REVIEWER, PhysicalScaleMorphismPartyRole.NATIVE_OWNER},
    )
    if any(pair.issubset(roles) for roles in role_by_actor.values() for pair in sensitive_pairs):
        return False
    forbidden: dict[PhysicalScaleMorphismPartyRole, frozenset[PhysicalScaleMorphismAccessClass]] = {
        PhysicalScaleMorphismPartyRole.NATIVE_OWNER: frozenset({PhysicalScaleMorphismAccessClass.PREDICTION}),
        PhysicalScaleMorphismPartyRole.MAP_TEAM: frozenset({PhysicalScaleMorphismAccessClass.EVALUATION_OUTCOME}),
        PhysicalScaleMorphismPartyRole.CONSTRUCT_REVIEWER: frozenset(
            {PhysicalScaleMorphismAccessClass.EVALUATION_OUTCOME, PhysicalScaleMorphismAccessClass.PREDICTION}
        ),
        PhysicalScaleMorphismPartyRole.METHOD_TEAM: frozenset({PhysicalScaleMorphismAccessClass.EVALUATION_OUTCOME}),
        PhysicalScaleMorphismPartyRole.LAB_TEAM: frozenset({PhysicalScaleMorphismAccessClass.PREDICTION}),
        PhysicalScaleMorphismPartyRole.CUSTODIAN: frozenset({PhysicalScaleMorphismAccessClass.EVALUATION_OUTCOME}),
    }
    for event in events:
        roles = role_by_actor.get(event.actor_id)
        if roles is None:
            return False
        if any(event.access_class in forbidden.get(role, frozenset()) for role in roles):
            if event.access_class is not PhysicalScaleMorphismAccessClass.PREDICTION or event.before_dossier_lock:
                return False
        if event.access_class is PhysicalScaleMorphismAccessClass.EVALUATION_OUTCOME:
            if event.outcome_access not in {
                OutcomeAccess.EVALUATOR_REVEAL,
                OutcomeAccess.EVALUATION_REVEALED,
            }:
                return False
            if PhysicalScaleMorphismPartyRole.EVALUATOR not in roles:
                return False
    return True


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismContaminationLedger(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-contamination-ledger'

    ledger_id: str
    dossier_lock_event_id: str
    mapping_lock_event_id: str
    prediction_roster_lock_event_id: str
    evaluation_roster_lock_event_id: str
    role_assignments: tuple[PhysicalScaleMorphismRoleAssignment, ...]
    access_events: tuple[PhysicalScaleMorphismAccessEvent, ...]
    prior_rc_artifact_ids: tuple[str, ...]
    prediction_generation_code_prompt_document_ids: tuple[str, ...]
    maximum_evidence_ceiling: EvidenceCeiling
    separation_qualified: bool
    locked: bool

    def __post_init__(self) -> None:
        for name in (
            "ledger_id",
            "dossier_lock_event_id",
            "mapping_lock_event_id",
            "prediction_roster_lock_event_id",
            "evaluation_roster_lock_event_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.role_assignments, attribute="assignment_id", field_name="role_assignments"
        )
        actors = tuple(value.actor_id for value in self.role_assignments)
        if len(set(actors)) != len(actors):
            raise ValueError("contamination ledger repeats an actor assignment")
        covered_roles = tuple(
            sorted({role.value for value in self.role_assignments for role in value.roles})
        )
        if covered_roles != PHYSICAL_SCALE_MORPHISM_REQUIRED_PARTY_ROLES:
            raise ValueError("contamination ledger does not assign every required party role")
        require_sorted_unique_ids(
            self.access_events, attribute="event_id", field_name="access_events"
        )
        ordinals = tuple(value.event_ordinal for value in self.access_events)
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("contamination ledger repeats an event ordinal")
        require_sorted_unique_strings(
            self.prior_rc_artifact_ids, field_name="prior_rc_artifact_ids"
        )
        require_sorted_unique_strings(
            self.prediction_generation_code_prompt_document_ids,
            field_name="prediction_generation_code_prompt_document_ids",
            allow_empty=False,
        )
        expected = _access_separation_qualified(self.role_assignments, self.access_events)
        if self.separation_qualified != expected:
            raise ValueError("contamination-ledger separation is not event-derived")
        if not self.locked:
            raise ValueError("contamination ledger must be locked before canary entry")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismConstructQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-construct-qualification'

    qualification_id: str
    dossier: ObjectIdentity
    mapping: ObjectIdentity
    review: ObjectIdentity
    contamination_ledger: ObjectIdentity
    terminal: PhysicalScaleMorphismConstructTerminal
    maximum_evidence_ceiling: EvidenceCeiling
    source_profile_derivation_allowed: bool
    canary_authoring_eligible: bool
    physical_action_authorized: bool
    claim_promotion_allowed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        for value, schema, name in (
            (self.dossier, IndependentApparatusDossier.SCHEMA, "dossier"),
            (self.mapping, IndependentConstructMapping.SCHEMA, "mapping"),
            (self.review, ConstructIndependenceReview.SCHEMA, "review"),
            (self.contamination_ledger, PhysicalScaleMorphismContaminationLedger.SCHEMA, "contamination_ledger"),
        ):
            _require_identity(value, schema, field_name=name)
        require_sorted_unique_strings(
            self.reason_codes, field_name="reason_codes", allow_empty=False
        )
        qualified = self.terminal is PhysicalScaleMorphismConstructTerminal.CONSTRUCT_INDEPENDENCE_QUALIFIED
        if (
            self.source_profile_derivation_allowed != qualified
            or self.canary_authoring_eligible != qualified
        ):
            raise ValueError("construct downstream eligibility is not terminal-derived")
        if self.physical_action_authorized or self.claim_promotion_allowed:
            raise ValueError("IP-6 qualification grants neither physical authority nor a claim")


def qualify_construct_independence(
    *,
    qualification_id: str,
    dossier: IndependentApparatusDossier,
    mapping: IndependentConstructMapping,
    review: ConstructIndependenceReview,
    ledger: PhysicalScaleMorphismContaminationLedger,
) -> PhysicalScaleMorphismConstructQualification:
    """Adjudicate IP-6 without accepting labels, responses, or authority."""

    dossier_identity = ObjectIdentity.from_record(dossier.dossier_id, dossier)
    mapping_identity = ObjectIdentity.from_record(mapping.mapping_id, mapping)
    if mapping.dossier != dossier_identity:
        raise ValueError("construct mapping does not bind the supplied dossier")
    if review.dossier != dossier_identity or review.mapping != mapping_identity:
        raise ValueError("construct review does not bind the supplied dossier/mapping")
    if set(value.native_object_id for value in mapping.entries) != set(dossier.native_object_ids):
        raise ValueError("construct mapping does not cover the native object roster exactly")
    assignments = {value.actor_id: set(value.roles) for value in ledger.role_assignments}
    authors_are_native = all(
        PhysicalScaleMorphismPartyRole.NATIVE_OWNER in assignments.get(value, set())
        for value in dossier.native_author_ids
    )
    map_team_matches = set(mapping.map_team_ids) == {
        actor for actor, roles in assignments.items() if PhysicalScaleMorphismPartyRole.MAP_TEAM in roles
    }
    review_parties_match = (
        set(review.native_owner_ids) == set(dossier.native_author_ids)
        and set(review.map_team_ids) == set(mapping.map_team_ids)
        and PhysicalScaleMorphismPartyRole.CONSTRUCT_REVIEWER in assignments.get(review.reviewer_id, set())
    )
    qualified = all(
        (
            review.qualified,
            ledger.separation_qualified,
            authors_are_native,
            map_team_matches,
            review_parties_match,
            ledger.dossier_lock_event_id == dossier.lock_event_id,
        )
    )
    terminal = (
        PhysicalScaleMorphismConstructTerminal.CONSTRUCT_INDEPENDENCE_QUALIFIED
        if qualified
        else PhysicalScaleMorphismConstructTerminal.CONSTRUCT_INDEPENDENCE_NOT_QUALIFIED
    )
    reason_codes = (
        ("CONSTRUCT_INDEPENDENCE_QUALIFIED",)
        if qualified
        else tuple(
            sorted(
                {
                    *(() if review.qualified else ("CONSTRUCT_REVIEW_NOT_QUALIFIED",)),
                    *(() if ledger.separation_qualified else ("ACCESS_OR_ROLE_SEPARATION_FAILED",)),
                    *(() if authors_are_native else ("NATIVE_AUTHOR_ROLE_MISMATCH",)),
                    *(() if map_team_matches else ("MAP_TEAM_ROLE_MISMATCH",)),
                    *(() if review_parties_match else ("REVIEW_PARTY_ROSTER_MISMATCH",)),
                    *(
                        ()
                        if ledger.dossier_lock_event_id == dossier.lock_event_id
                        else ("DOSSIER_LOCK_EVENT_MISMATCH",)
                    ),
                }
            )
        )
    )
    return PhysicalScaleMorphismConstructQualification(
        qualification_id=qualification_id,
        dossier=dossier_identity,
        mapping=mapping_identity,
        review=ObjectIdentity.from_record(review.review_id, review),
        contamination_ledger=ObjectIdentity.from_record(ledger.ledger_id, ledger),
        terminal=terminal,
        maximum_evidence_ceiling=ledger.maximum_evidence_ceiling,
        source_profile_derivation_allowed=qualified,
        canary_authoring_eligible=qualified,
        physical_action_authorized=False,
        claim_promotion_allowed=False,
        reason_codes=reason_codes,
    )


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismApparatusGateAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-apparatus-gate-assessment'

    gate_id: str
    state: PhysicalScaleMorphismGateState
    evidence_artifact_locator_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.gate_id, field_name="gate_id")
        if self.gate_id not in PHYSICAL_SCALE_MORPHISM_APPARATUS_GATE_IDS:
            raise ValueError("apparatus qualification contains an unknown gate")
        require_sorted_unique_strings(
            self.evidence_artifact_locator_ids,
            field_name="evidence_artifact_locator_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.state is PhysicalScaleMorphismGateState.PASS and self.reason_codes:
            raise ValueError("passing apparatus gate cannot retain failure reasons")
        if self.state is not PhysicalScaleMorphismGateState.PASS and not self.reason_codes:
            raise ValueError("nonpassing apparatus gate requires reason codes")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismSourceApparatusQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-source-apparatus-qualification'

    qualification_id: str
    construct_qualification: ObjectIdentity
    semantic_profile: ObjectIdentity
    excluded_canary_board_ids: tuple[str, ...]
    gate_assessments: tuple[PhysicalScaleMorphismApparatusGateAssessment, ...]
    physical_authority_record_ids: tuple[str, ...]
    publication_receipt_ids: tuple[str, ...]
    recovery_replay_passed: bool
    canary_evidence_promotable: bool
    terminal: PhysicalScaleMorphismSourceApparatusTerminal
    frozen: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        _require_identity(
            self.construct_qualification,
            PhysicalScaleMorphismConstructQualification.SCHEMA,
            field_name="construct_qualification",
        )
        _require_identity(
            self.semantic_profile,
            'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-physical-semantic-profile',
            field_name="semantic_profile",
        )
        require_sorted_unique_strings(
            self.excluded_canary_board_ids,
            field_name="excluded_canary_board_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.gate_assessments, attribute="gate_id", field_name="gate_assessments"
        )
        if tuple(value.gate_id for value in self.gate_assessments) != PHYSICAL_SCALE_MORPHISM_APPARATUS_GATE_IDS:
            raise ValueError("source/apparatus qualification does not cover every frozen gate")
        require_sorted_unique_strings(
            self.physical_authority_record_ids,
            field_name="physical_authority_record_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.publication_receipt_ids, field_name="publication_receipt_ids", allow_empty=False
        )
        expected = (
            all(value.state is PhysicalScaleMorphismGateState.PASS for value in self.gate_assessments)
            and self.recovery_replay_passed
        )
        if (
            self.terminal is PhysicalScaleMorphismSourceApparatusTerminal.SOURCE_AND_APPARATUS_QUALIFIED
        ) != expected:
            raise ValueError("source/apparatus terminal is not gate-derived")
        if self.canary_evidence_promotable:
            raise ValueError("excluded canary evidence is permanently nonpromotable")
        if not self.frozen:
            raise ValueError("source/apparatus qualification must be immutable")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismDevelopmentHandoff(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-development-handoff'

    handoff_id: str
    source_apparatus_qualification: ObjectIdentity
    development_board_ids: tuple[str, ...]
    development_batch_ids: tuple[str, ...]
    contacted_scale_cells: tuple[int, ...]
    n64_response_count: int
    selected_coordinate_ids: tuple[str, ...]
    selected_receiver_id: str
    selected_history_id: str
    selected_numerical_model_id: str
    retained_comparator_ids: tuple[str, ...]
    power_qualification: PhysicalScaleMorphismPowerQualification
    morphism_domain_complete: bool
    equivalence_rule_id: str
    boundary_rule_id: str
    multiplicity_rule_id: str
    obstruction_set: ObjectIdentity
    development_authority_record_ids: tuple[str, ...]
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess
    frozen: bool

    def __post_init__(self) -> None:
        for name in (
            "handoff_id",
            "selected_receiver_id",
            "selected_history_id",
            "selected_numerical_model_id",
            "equivalence_rule_id",
            "boundary_rule_id",
            "multiplicity_rule_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        _require_identity(
            self.source_apparatus_qualification,
            PhysicalScaleMorphismSourceApparatusQualification.SCHEMA,
            field_name="source_apparatus_qualification",
        )
        _require_identity(
            self.obstruction_set,
            PhysicalScaleMorphismObstructionSet.SCHEMA,
            field_name="obstruction_set",
        )
        require_sorted_unique_strings(
            self.development_board_ids, field_name="development_board_ids", allow_empty=False
        )
        require_sorted_unique_strings(
            self.development_batch_ids, field_name="development_batch_ids", allow_empty=False
        )
        if self.contacted_scale_cells != (16, 32) or self.n64_response_count:
            raise ValueError("IP-8 development must contact N16/N32 and keep N64 held out")
        require_sorted_unique_strings(
            self.selected_coordinate_ids,
            field_name="selected_coordinate_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.retained_comparator_ids,
            field_name="retained_comparator_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.development_authority_record_ids,
            field_name="development_authority_record_ids",
            allow_empty=False,
        )
        if self.evaluation_outcome_count:
            raise ValueError("development handoff cannot contain evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("development handoff must retain development visibility")
        if not self.frozen:
            raise ValueError("development handoff must be frozen before evaluation authoring")

    @property
    def power_terminal(self) -> PhysicalScaleMorphismPowerTerminal:
        return self.power_qualification.terminal


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismEvaluationScaleRoster(CanonicalRecord):
    """Fresh evaluation boards and fabrication batches at one physical scale."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-evaluation-scale-roster'

    roster_id: str
    scale_cells: int
    board_ids: tuple[str, ...]
    batch_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        if self.scale_cells not in PHYSICAL_SCALE_MORPHISM_PHYSICAL_SCALE_CELLS:
            raise ValueError("evaluation scale roster contains an unsupported scale")
        require_sorted_unique_strings(self.board_ids, field_name="board_ids", allow_empty=False)
        require_sorted_unique_strings(self.batch_ids, field_name="batch_ids", allow_empty=False)
        if len(self.batch_ids) < 2:
            raise ValueError("entered evaluation scale requires at least two fabrication batches")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismPredictionFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-prediction-freeze'

    freeze_id: str
    plan_sha256: str
    construct_qualification: ObjectIdentity
    source_apparatus_qualification: ObjectIdentity
    development_handoff: ObjectIdentity
    frozen_object_identities: tuple[ObjectIdentity, ...]
    evaluation_scale_rosters: tuple[PhysicalScaleMorphismEvaluationScaleRoster, ...]
    board_count_per_scale: int
    reserve_rule_id: str
    assignment_seed: int
    candidate: ObjectIdentity
    execution_graph: ObjectIdentity
    evaluator_id: str
    authority_requirement_ids: tuple[str, ...]
    protected_evaluation_outcome_count_at_freeze: int
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    frozen: bool

    def __post_init__(self) -> None:
        for name in ("freeze_id", "reserve_rule_id", "evaluator_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.plan_sha256, field_name="plan_sha256")
        _require_identity(
            self.construct_qualification,
            PhysicalScaleMorphismConstructQualification.SCHEMA,
            field_name="construct_qualification",
        )
        _require_identity(
            self.source_apparatus_qualification,
            PhysicalScaleMorphismSourceApparatusQualification.SCHEMA,
            field_name="source_apparatus_qualification",
        )
        _require_identity(
            self.development_handoff,
            PhysicalScaleMorphismDevelopmentHandoff.SCHEMA,
            field_name="development_handoff",
        )
        object_keys = tuple(
            (value.object_schema, value.object_id, value.object_fingerprint)
            for value in self.frozen_object_identities
        )
        if tuple(sorted(set(object_keys))) != object_keys or not object_keys:
            raise ValueError("prediction freeze object roster must be sorted, unique and nonempty")
        observed_schemas = {value.object_schema for value in self.frozen_object_identities}
        missing_schemas = set(PHYSICAL_SCALE_MORPHISM_PREDICTION_FREEZE_REQUIRED_OBJECT_SCHEMAS) - observed_schemas
        if missing_schemas:
            raise ValueError(
                f"prediction freeze lacks required object schemas: {sorted(missing_schemas)}"
            )
        normalization_uncertainties = tuple(
            value
            for value in self.frozen_object_identities
            if value.object_schema == 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-normalization-uncertainty'
        )
        if len(normalization_uncertainties) != 1:
            raise ValueError(
                "prediction freeze requires exactly one normalization-uncertainty identity"
            )
        require_sorted_unique_ids(
            self.evaluation_scale_rosters,
            attribute="roster_id",
            field_name="evaluation_scale_rosters",
        )
        if tuple(value.scale_cells for value in self.evaluation_scale_rosters) != (
            PHYSICAL_SCALE_MORPHISM_PHYSICAL_SCALE_CELLS
        ):
            raise ValueError("prediction freeze requires the complete N16/N32/N64 scale roster")
        if self.board_count_per_scale not in {8, 12, 16, 24}:
            raise ValueError("prediction freeze board count lies outside the power ladder")
        if any(
            len(value.board_ids) != self.board_count_per_scale
            for value in self.evaluation_scale_rosters
        ):
            raise ValueError("prediction freeze does not use one common per-scale board count")
        all_board_ids = tuple(
            board_id for value in self.evaluation_scale_rosters for board_id in value.board_ids
        )
        if len(set(all_board_ids)) != len(all_board_ids):
            raise ValueError("prediction freeze repeats an evaluation board across scales")
        if self.assignment_seed < 0:
            raise ValueError("assignment seed must be nonnegative")
        if self.candidate.object_schema not in {
            'empirical-lawhood/runtime/draft-study-candidate',
            'empirical-lawhood/runtime/study-candidate',
        }:
            raise ValueError("prediction freeze binds the wrong candidate schema")
        if self.execution_graph.object_schema != 'empirical-lawhood/runtime/candidate-execution-plan':
            raise ValueError("prediction freeze binds the wrong execution-graph schema")
        require_sorted_unique_strings(
            self.authority_requirement_ids,
            field_name="authority_requirement_ids",
            allow_empty=False,
        )
        if self.protected_evaluation_outcome_count_at_freeze:
            raise ValueError("prediction freeze occurred after evaluation outcome contact")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("prediction freeze must be authored outcome blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("prediction freeze must retain prospective visibility")
        if not self.frozen:
            raise ValueError("prediction freeze must be immutable")

    @property
    def evaluation_board_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                board_id for value in self.evaluation_scale_rosters for board_id in value.board_ids
            )
        )

    @property
    def evaluation_batch_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    batch_id
                    for value in self.evaluation_scale_rosters
                    for batch_id in value.batch_ids
                }
            )
        )

    @property
    def evaluation_scale_cells(self) -> tuple[int, ...]:
        return tuple(value.scale_cells for value in self.evaluation_scale_rosters)

    @property
    def uncertainty_method_id(self) -> str:
        return next(
            value.object_id
            for value in self.frozen_object_identities
            if value.object_schema == 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-normalization-uncertainty'
        )


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismIssueRecord(CanonicalRecord):
    """Outcome-blind IP-10 issue and fresh-board qualification handoff."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-issue-record'

    issue_id: str
    prediction_freeze: ObjectIdentity
    issued_candidate: ObjectIdentity
    issued_run: ObjectIdentity
    board_identity_records: tuple[ObjectIdentity, ...]
    pre_response_metrology_records: tuple[ObjectIdentity, ...]
    board_batch_binding_records: tuple[ObjectIdentity, ...]
    expected_board_count: int
    qualified_board_ids: tuple[str, ...]
    qualification_failed_board_ids: tuple[str, ...]
    reserve_substitution_ids: tuple[str, ...]
    board_specific_numerical_twin_ids: tuple[str, ...]
    issue_and_fabrication_authority_record_ids: tuple[str, ...]
    qualification_receipt_ids: tuple[str, ...]
    protected_response_count_at_issue: int
    outcome_access: OutcomeAccess
    terminal: PhysicalScaleMorphismIssueTerminal
    frozen: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.issue_id, field_name="issue_id")
        _require_identity(
            self.prediction_freeze, PhysicalScaleMorphismPredictionFreeze.SCHEMA, field_name="prediction_freeze"
        )
        if self.issued_run.object_schema != 'empirical-lawhood/runtime/candidate-run-plan':
            raise ValueError("IP-10 issue binds the wrong run-plan schema")
        for name in (
            "board_identity_records",
            "pre_response_metrology_records",
            "board_batch_binding_records",
        ):
            values = getattr(self, name)
            keys = tuple(
                (value.object_schema, value.object_id, value.object_fingerprint) for value in values
            )
            if tuple(sorted(set(keys))) != keys or not keys:
                raise ValueError(f"{name} must be sorted, unique and nonempty")
        if any(
            value.object_schema != 'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-board-identity'
            for value in self.board_identity_records
        ):
            raise ValueError("IP-10 board roster contains a non-board identity")
        if any(
            value.object_schema != 'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-component-metrology'
            for value in self.pre_response_metrology_records
        ):
            raise ValueError("IP-10 metrology roster contains a non-metrology identity")
        if any(
            value.object_schema != 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-board-batch-binding'
            for value in self.board_batch_binding_records
        ):
            raise ValueError("IP-10 batch-linkage roster contains another record type")
        for name in (
            "qualified_board_ids",
            "qualification_failed_board_ids",
            "reserve_substitution_ids",
            "board_specific_numerical_twin_ids",
            "issue_and_fabrication_authority_record_ids",
            "qualification_receipt_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name in {"qualification_failed_board_ids", "reserve_substitution_ids"},
            )
        if set(self.qualified_board_ids) & set(self.qualification_failed_board_ids):
            raise ValueError("a board cannot both pass and fail pre-response qualification")
        if self.expected_board_count <= 0:
            raise ValueError("IP-10 expected board count must be positive")
        if set(self.qualified_board_ids) != {
            value.object_id for value in self.board_identity_records
        }:
            raise ValueError("qualified board IDs differ from the final board-identity roster")
        if len(self.board_identity_records) != len(self.pre_response_metrology_records):
            raise ValueError("every entered board requires one pre-response metrology record")
        if len(self.board_identity_records) != len(self.board_batch_binding_records):
            raise ValueError("every entered board requires one immutable batch-linkage record")
        if len(self.board_specific_numerical_twin_ids) != len(self.qualified_board_ids):
            raise ValueError("every qualified board requires one frozen numerical twin")
        if self.protected_response_count_at_issue:
            raise ValueError("IP-10 issue occurred after protected response contact")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("IP-10 issue and qualification must remain outcome blind")
        expected_complete = len(self.qualified_board_ids) == self.expected_board_count
        if (self.terminal is PhysicalScaleMorphismIssueTerminal.ISSUED_COMPLETE_ROSTER) != expected_complete:
            raise ValueError("issue terminal is not derived from the qualified board roster")
        if not self.frozen:
            raise ValueError("IP-10 issue record must be immutable")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismExecutionSeal(CanonicalRecord):
    """Sealed IP-11 physical/numerical execution roster before evaluator fan-in."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-execution-seal'

    seal_id: str
    prediction_freeze: ObjectIdentity
    issue_record: ObjectIdentity
    run: ObjectIdentity
    expected_unit_ids: tuple[str, ...]
    sealed_bundle_unit_ids: tuple[str, ...]
    typed_terminal_unit_ids: tuple[str, ...]
    physical_bundle_artifact_ids: tuple[str, ...]
    numerical_artifact_ids: tuple[str, ...]
    publication_receipt_ids: tuple[str, ...]
    uncertain_delivery_unit_ids: tuple[str, ...]
    reconciliation_record_ids: tuple[str, ...]
    physical_execution_authority_record_ids: tuple[str, ...]
    laboratory_prediction_label_access_count: int
    automatic_physical_retry_count: int
    complete: bool
    outcome_access: OutcomeAccess
    sealed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.seal_id, field_name="seal_id")
        _require_identity(
            self.prediction_freeze, PhysicalScaleMorphismPredictionFreeze.SCHEMA, field_name="prediction_freeze"
        )
        _require_identity(self.issue_record, PhysicalScaleMorphismIssueRecord.SCHEMA, field_name="issue_record")
        if self.run.object_schema != 'empirical-lawhood/runtime/candidate-run-plan':
            raise ValueError("IP-11 execution binds the wrong run-plan schema")
        for name in (
            "expected_unit_ids",
            "sealed_bundle_unit_ids",
            "typed_terminal_unit_ids",
            "physical_bundle_artifact_ids",
            "numerical_artifact_ids",
            "publication_receipt_ids",
            "uncertain_delivery_unit_ids",
            "reconciliation_record_ids",
            "physical_execution_authority_record_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name
                in {
                    "typed_terminal_unit_ids",
                    "uncertain_delivery_unit_ids",
                    "reconciliation_record_ids",
                },
            )
        if set(self.sealed_bundle_unit_ids) & set(self.typed_terminal_unit_ids):
            raise ValueError("execution unit cannot be both sealed evidence and a terminal")
        covered = set(self.sealed_bundle_unit_ids) | set(self.typed_terminal_unit_ids)
        expected_complete = covered == set(self.expected_unit_ids)
        if self.complete != expected_complete:
            raise ValueError("execution completeness is not exact-roster derived")
        if not set(self.uncertain_delivery_unit_ids).issubset(self.expected_unit_ids):
            raise ValueError("uncertain delivery contains an unexpected execution unit")
        if self.uncertain_delivery_unit_ids and not self.reconciliation_record_ids:
            raise ValueError("uncertain physical delivery requires explicit reconciliation")
        if (
            self.laboratory_prediction_label_access_count < 0
            or self.automatic_physical_retry_count < 0
        ):
            raise ValueError("execution access/retry counts must be nonnegative")
        if self.laboratory_prediction_label_access_count:
            raise ValueError("laboratory execution crossed the expected-label firewall")
        if self.automatic_physical_retry_count:
            raise ValueError("physical actions cannot be retried automatically")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("IP-11 execution bundles must remain sealed")
        if not self.sealed:
            raise ValueError("IP-11 execution handoff must be sealed before fan-in")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismCompleteFanIn(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-complete-fan-in'

    fan_in_id: str
    prediction_freeze: ObjectIdentity
    execution_seal: ObjectIdentity
    expected_unit_ids: tuple[str, ...]
    received_unit_ids: tuple[str, ...]
    terminal_unit_ids: tuple[str, ...]
    publication_receipt_ids: tuple[str, ...]
    recovery_event_ids: tuple[str, ...]
    recovery_proof_ids: tuple[str, ...]
    sealed_payload_artifact_ids: tuple[str, ...]
    outcome_access_event_ids: tuple[str, ...]
    unexpected_unit_ids: tuple[str, ...]
    complete: bool
    outcome_access: OutcomeAccess
    frozen: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.fan_in_id, field_name="fan_in_id")
        _require_identity(
            self.prediction_freeze, PhysicalScaleMorphismPredictionFreeze.SCHEMA, field_name="prediction_freeze"
        )
        _require_identity(
            self.execution_seal, PhysicalScaleMorphismExecutionSeal.SCHEMA, field_name="execution_seal"
        )
        for name in (
            "expected_unit_ids",
            "received_unit_ids",
            "terminal_unit_ids",
            "publication_receipt_ids",
            "recovery_event_ids",
            "recovery_proof_ids",
            "sealed_payload_artifact_ids",
            "outcome_access_event_ids",
            "unexpected_unit_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name
                in {"terminal_unit_ids", "recovery_event_ids", "unexpected_unit_ids"},
            )
        covered = set(self.received_unit_ids) | set(self.terminal_unit_ids)
        expected_complete = covered == set(self.expected_unit_ids) and not self.unexpected_unit_ids
        if self.complete != expected_complete:
            raise ValueError("complete fan-in is not exact-roster derived")
        if set(self.received_unit_ids) & set(self.terminal_unit_ids):
            raise ValueError("fan-in unit cannot be both received and terminal")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("fan-in record must be frozen before evaluator reveal")
        if not self.frozen:
            raise ValueError("fan-in record must be immutable before reveal")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismScaleSimulationCertificate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-scale-simulation-certificate'

    certificate_id: str
    morphism_id: str
    source_scale_cells: int
    target_scale_cells: int
    property_ids: tuple[str, ...]
    directed: bool
    conservative: bool
    reverse_claimed: bool
    state: PhysicalScaleMorphismSimulationState
    reason_codes: tuple[str, ...]
    evidence_world: PhysicalScaleMorphismEvidenceWorld
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        for name in ("certificate_id", "morphism_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.source_scale_cells not in PHYSICAL_SCALE_MORPHISM_PHYSICAL_SCALE_CELLS
            or self.target_scale_cells not in PHYSICAL_SCALE_MORPHISM_PHYSICAL_SCALE_CELLS
        ):
            raise ValueError("scale-simulation certificate lies outside the physical scale roster")
        if self.source_scale_cells >= self.target_scale_cells:
            raise ValueError(
                "scale-simulation certificate must follow the frozen small-to-large direction"
            )
        require_sorted_unique_strings(
            self.property_ids, field_name="property_ids", allow_empty=False
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not self.directed or self.reverse_claimed:
            raise ValueError("scale simulation is directed and cannot imply a reverse map")
        if self.state is PhysicalScaleMorphismSimulationState.SUPPORTED and (
            not self.conservative or self.reason_codes
        ):
            raise ValueError("supported scale simulation must be conservative and reason-free")
        if self.state is not PhysicalScaleMorphismSimulationState.SUPPORTED and not self.reason_codes:
            raise ValueError("non-supported scale simulation requires reason codes")
        if self.evidence_world is not PhysicalScaleMorphismEvidenceWorld.PHYSICAL_RC:
            raise ValueError(
                "physical scale certificate cannot be issued from another evidence world"
            )
        if self.evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("physical scale morphism scale simulation has the frozen admission ceiling")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismImplementationSignature(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-implementation-signature'

    signature_id: str
    implementation_id: str
    complete_scale_cells: tuple[int, ...]
    terminal_ids: tuple[str, ...]
    signature_sha256: str

    def __post_init__(self) -> None:
        for name in ("signature_id", "implementation_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.complete_scale_cells not in {(16, 32), PHYSICAL_SCALE_MORPHISM_PHYSICAL_SCALE_CELLS}:
            raise ValueError("implementation signature lacks a complete two/three-scale chain")
        require_sorted_unique_strings(
            self.terminal_ids, field_name="terminal_ids", allow_empty=False
        )
        validate_sha256(self.signature_sha256, field_name="signature_sha256")
        if self.signature_sha256 != implementation_signature_sha256(
            complete_scale_cells=self.complete_scale_cells,
            terminal_ids=self.terminal_ids,
        ):
            raise ValueError("implementation signature hash is not terminal/scale derived")


def implementation_signature_sha256(
    *,
    complete_scale_cells: tuple[int, ...],
    terminal_ids: tuple[str, ...],
) -> str:
    """Hash only the compact nonpooled comparison signature, not lab identity."""

    return sha256(
        canonical_json_bytes(
            {
                "complete_scale_cells": complete_scale_cells,
                "terminal_ids": terminal_ids,
            }
        )
    ).hexdigest()


def make_implementation_signature(
    *,
    signature_id: str,
    implementation_id: str,
    complete_scale_cells: tuple[int, ...],
    terminal_ids: tuple[str, ...],
) -> PhysicalScaleMorphismImplementationSignature:
    return PhysicalScaleMorphismImplementationSignature(
        signature_id=signature_id,
        implementation_id=implementation_id,
        complete_scale_cells=complete_scale_cells,
        terminal_ids=terminal_ids,
        signature_sha256=implementation_signature_sha256(
            complete_scale_cells=complete_scale_cells,
            terminal_ids=terminal_ids,
        ),
    )


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismImplementationRecurrencePanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-implementation-recurrence-panel'

    panel_id: str
    entered_implementation_ids: tuple[str, ...]
    signatures: tuple[PhysicalScaleMorphismImplementationSignature, ...]
    pooled_native_coefficients: bool
    pooled_thresholds_or_likelihoods: bool
    state: PhysicalScaleMorphismImplementationRecurrenceState
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        require_sorted_unique_strings(
            self.entered_implementation_ids,
            field_name="entered_implementation_ids",
            allow_empty=False,
        )
        if len(self.entered_implementation_ids) not in {1, 2}:
            raise ValueError("physical scale morphism enters one base or two independent implementations")
        require_sorted_unique_ids(
            self.signatures, attribute="signature_id", field_name="signatures"
        )
        signature_implementations = tuple(value.implementation_id for value in self.signatures)
        if tuple(sorted(signature_implementations)) != signature_implementations:
            raise ValueError("implementation signatures must be ordered by implementation")
        if not set(signature_implementations).issubset(self.entered_implementation_ids):
            raise ValueError("recurrence panel contains an unentered implementation")
        if self.pooled_native_coefficients or self.pooled_thresholds_or_likelihoods:
            raise ValueError("independent implementations cannot pool native science")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.state is PhysicalScaleMorphismImplementationRecurrenceState.SUPPORTED:
            if len(self.entered_implementation_ids) != 2 or len(self.signatures) != 2:
                raise ValueError(
                    "supported independent recurrence requires two complete signatures"
                )
            if len({value.signature_sha256 for value in self.signatures}) != 1:
                raise ValueError(
                    "supported independent recurrence requires the same compact signature"
                )
            if self.reason_codes:
                raise ValueError("supported implementation recurrence cannot retain reasons")
        elif not self.reason_codes:
            raise ValueError("non-supported implementation recurrence requires reason codes")
        if (
            len(self.entered_implementation_ids) == 1
            and self.state is not PhysicalScaleMorphismImplementationRecurrenceState.NOT_TESTED
        ):
            raise ValueError("one implementation cannot adjudicate independent recurrence")


__all__ = [
    "ConstructIndependenceReview",
    "PHYSICAL_SCALE_MORPHISM_APPARATUS_GATE_IDS",
    "PHYSICAL_SCALE_MORPHISM_CONSTRUCT_REVIEW_QUESTION_IDS",
    "PHYSICAL_SCALE_MORPHISM_NATIVE_DOSSIER_SECTION_IDS",
    "PHYSICAL_SCALE_MORPHISM_PHYSICAL_SCALE_CELLS",
    "PHYSICAL_SCALE_MORPHISM_PREDICTION_FREEZE_REQUIRED_OBJECT_SCHEMAS",
    "PHYSICAL_SCALE_MORPHISM_REQUIRED_PARTY_ROLES",
    'PhysicalScaleMorphismAccessClass',
    'PhysicalScaleMorphismAccessEvent',
    'PhysicalScaleMorphismApparatusGateAssessment',
    'PhysicalScaleMorphismCompleteFanIn',
    'PhysicalScaleMorphismConstructMapEntry',
    'PhysicalScaleMorphismConstructQualification',
    'PhysicalScaleMorphismConstructReviewItem',
    'PhysicalScaleMorphismConstructTerminal',
    'PhysicalScaleMorphismContaminationLedger',
    'PhysicalScaleMorphismDevelopmentHandoff',
    'PhysicalScaleMorphismEvaluationScaleRoster',
    'PhysicalScaleMorphismGateState',
    'PhysicalScaleMorphismImplementationRecurrencePanel',
    'PhysicalScaleMorphismImplementationRecurrenceState',
    'PhysicalScaleMorphismImplementationSignature',
    'PhysicalScaleMorphismIssueRecord',
    'PhysicalScaleMorphismIssueTerminal',
    'PhysicalScaleMorphismPartyRole',
    'PhysicalScaleMorphismPredictionFreeze',
    'PhysicalScaleMorphismReviewAnswer',
    'PhysicalScaleMorphismScaleSimulationCertificate',
    'PhysicalScaleMorphismExecutionSeal',
    'PhysicalScaleMorphismSimulationState',
    'PhysicalScaleMorphismSourceApparatusQualification',
    'PhysicalScaleMorphismSourceApparatusTerminal',
    'PhysicalScaleMorphismRoleAssignment',
    "IndependentApparatusDossier",
    "IndependentConstructMapping",
    "qualify_construct_independence",
    "implementation_signature_sha256",
    "make_implementation_signature",
]
