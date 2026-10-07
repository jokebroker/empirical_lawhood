"""Pure response-algebra witnesses required before controller construction.

These additive records bind causal support and prepared-prefix composition to
the accepted ``L(D,H,A,R,tau)`` identity.  They contain compact categorical
evidence and coordinates only; trajectories and fitted payloads remain behind
external evidence links.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from .decoding import decode_canonical_bytes
from .evidence import (
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from .laws import ResponseLaw
from .obligations import ObligationStatus
from .provenance import EvidenceLink, ObjectIdentity
from .references import ExecutableReference
from .response_algebra import LetterActionWord
from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)
from .systems import RelationalIdentity


MAX_RESPONSE_ALGEBRA_CONTROL_RECORD_BYTES = 1024 * 1024


class CausalSupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    UNEVALUABLE = "UNEVALUABLE"


class TemporalPredicateSemantics(StrEnum):
    ALWAYS_PRESERVED_PATH = "ALWAYS_PRESERVED_PATH"
    TERMINAL = "TERMINAL"
    INTERVAL_INTEGRAL = "INTERVAL_INTEGRAL"
    EVENTUALLY_REACHED = "EVENTUALLY_REACHED"
    RECOVERABLE_WINDOW = "RECOVERABLE_WINDOW"


class PrefixCompositionStatus(StrEnum):
    DEFINED = "DEFINED"
    PREFIX_OUTSIDE_SUPPORT = "PREFIX_OUTSIDE_SUPPORT"
    OBLIGATION_FAILED_BEFORE_INFLUENCE = "OBLIGATION_FAILED_BEFORE_INFLUENCE"
    ACTION_CAUSAL_CONE_MISSING = "ACTION_CAUSAL_CONE_MISSING"
    OPPOSED = "OPPOSED"
    UNEVALUABLE = "UNEVALUABLE"


_REASON_PREFIXES = (
    "ACTION_",
    "CLOCK_",
    "EVIDENCE_",
    "OBLIGATION_",
    "OUTCOME_",
    "PREFIX_",
    "PREREQUISITE_",
    "RECEIVER_",
    "SUPPORT_",
)


def _validate_reason_codes(values: tuple[str, ...]) -> None:
    require_sorted_unique_strings(values, field_name="reason_codes")
    if any(not value.startswith(_REASON_PREFIXES) for value in values):
        raise ValueError("response-algebra control reason is outside the frozen families")


def _validate_visibility(
    *,
    evidence_ceiling: EvidenceCeiling,
    outcome_access: OutcomeAccess,
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...],
    visibility_ceiling: VisibilityCeiling,
    evidence_links: tuple[EvidenceLink, ...],
) -> None:
    inherited = inherited_visibility(
        (*parent_visibility_ceilings, *(link.visibility_ceiling for link in evidence_links)),
        outcome_access,
    )
    if not visibility_ceiling.is_at_least_as_restrictive_as(inherited):
        raise ValueError("response-algebra control visibility cannot be lowered")
    if not visibility_ceiling.is_promotable:
        if evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("outcome-visible response-algebra control evidence is non-promotable")


def _retains_categorical_evidence_at(
    evidence_ceiling: EvidenceCeiling,
    rung: EvidenceRung,
) -> bool:
    """Allow honest non-promotable findings without allowing their promotion."""

    return evidence_ceiling.allows(rung) or evidence_ceiling is EvidenceCeiling.NON_PROMOTABLE


@dataclass(frozen=True, slots=True)
class CausalSupportAssessment(CanonicalRecord):
    """Evidence that a delivered word has one bounded future causal cone."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/causal-support-assessment'

    assessment_id: str
    relation: RelationalIdentity
    world_id: str
    action_word_id: str
    denominator_id: str
    retained_history_id: str
    receiver_id: str
    horizon_id: str
    requested_clock_id: str
    requested_receiver_coordinate: Decimal
    applied_clock_id: str
    applied_receiver_coordinate: Decimal
    realized_clock_id: str
    realized_receiver_coordinate: Decimal
    receiver_clock_id: str
    clock_relation_id: str | None
    earliest_supported_receiver_coordinate: Decimal | None
    latest_supported_receiver_coordinate: Decimal | None
    status: CausalSupportStatus
    physical_independent_unit_id: str
    physical_independent_unit_count: int
    numerical_view_ids: tuple[str, ...]
    evidence_links: tuple[EvidenceLink, ...]
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("world_id", self.world_id),
            ("action_word_id", self.action_word_id),
            ("denominator_id", self.denominator_id),
            ("retained_history_id", self.retained_history_id),
            ("receiver_id", self.receiver_id),
            ("horizon_id", self.horizon_id),
            ("requested_clock_id", self.requested_clock_id),
            ("applied_clock_id", self.applied_clock_id),
            ("realized_clock_id", self.realized_clock_id),
            ("receiver_clock_id", self.receiver_clock_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.clock_relation_id is not None:
            validate_stable_id(self.clock_relation_id, field_name="clock_relation_id")
        action_clocks = {
            self.requested_clock_id,
            self.applied_clock_id,
            self.realized_clock_id,
        }
        if action_clocks == {self.receiver_clock_id}:
            if self.clock_relation_id is not None:
                raise ValueError("a common clock cannot carry a cross-clock relation")
        elif self.clock_relation_id is None:
            raise ValueError("different action/receiver clocks require an exact clock relation")
        for name, coordinate in (
            ("requested_receiver_coordinate", self.requested_receiver_coordinate),
            ("applied_receiver_coordinate", self.applied_receiver_coordinate),
            ("realized_receiver_coordinate", self.realized_receiver_coordinate),
        ):
            validate_decimal(coordinate, field_name=name)
        if not (
            self.requested_receiver_coordinate
            <= self.applied_receiver_coordinate
            <= self.realized_receiver_coordinate
        ):
            raise ValueError("requested, applied and realized causal clocks are out of order")
        if (self.earliest_supported_receiver_coordinate is None) != (
            self.latest_supported_receiver_coordinate is None
        ):
            raise ValueError("causal support interval requires both endpoints")
        earliest = self.earliest_supported_receiver_coordinate
        latest = self.latest_supported_receiver_coordinate
        if earliest is not None:
            if latest is None:
                raise ValueError("causal support interval requires both endpoints")
            validate_decimal(
                earliest,
                field_name="earliest_supported_receiver_coordinate",
            )
            validate_decimal(
                latest,
                field_name="latest_supported_receiver_coordinate",
            )
            if earliest <= self.realized_receiver_coordinate:
                raise ValueError("receiver influence must be strictly after realized action")
            if latest < earliest:
                raise ValueError("causal support interval endpoints are reversed")
        if self.horizon_id != self.relation.horizon.horizon_id:
            raise ValueError("causal support horizon differs from its relation")
        if self.receiver_id not in self.relation.receiver_quantity_ids:
            raise ValueError("causal support receiver is outside its relation")
        if self.physical_independent_unit_count < 0:
            raise ValueError("physical independent-unit count must be nonnegative")
        require_sorted_unique_strings(
            self.numerical_view_ids,
            field_name="numerical_view_ids",
            allow_empty=False,
        )
        for view_id in self.numerical_view_ids:
            validate_stable_id(view_id, field_name="numerical_view_ids")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        if any(link.world_id != self.world_id for link in self.evidence_links):
            raise ValueError("causal support evidence crosses evidence worlds")
        _validate_reason_codes(self.reason_codes)
        if self.status is CausalSupportStatus.SUPPORTED:
            if self.earliest_supported_receiver_coordinate is None:
                raise ValueError("supported causal response requires an influence interval")
            if (
                self.physical_independent_unit_count == 0
                or not self.evidence_links
                or self.reason_codes
                or not _retains_categorical_evidence_at(
                    self.evidence_ceiling,
                    EvidenceRung.RESPONSE,
                )
            ):
                raise ValueError("supported causal response lacks categorical response evidence")
        elif self.status is CausalSupportStatus.OPPOSED:
            if self.earliest_supported_receiver_coordinate is not None:
                raise ValueError("opposed causal response cannot claim a supported interval")
            if (
                self.physical_independent_unit_count == 0
                or not self.evidence_links
                or not self.reason_codes
                or not _retains_categorical_evidence_at(
                    self.evidence_ceiling,
                    EvidenceRung.RESPONSE,
                )
            ):
                raise ValueError(
                    "opposed causal response requires categorical response evidence and reasons"
                )
        elif not self.reason_codes:
            raise ValueError("unevaluable causal response requires reasons")
        _validate_visibility(
            evidence_ceiling=self.evidence_ceiling,
            outcome_access=self.outcome_access,
            parent_visibility_ceilings=self.parent_visibility_ceilings,
            visibility_ceiling=self.visibility_ceiling,
            evidence_links=self.evidence_links,
        )


@dataclass(frozen=True, slots=True)
class PrefixObligationAssessment(CanonicalRecord):
    """Frozen temporal semantics and prefix-time disposition of one obligation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/prefix-obligation-assessment'

    obligation_id: str
    semantics: TemporalPredicateSemantics
    evaluator: ExecutableReference
    criterion: ObjectIdentity
    status: ObligationStatus
    evaluated_through_receiver_coordinate: Decimal
    first_failure_receiver_coordinate: Decimal | None
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.obligation_id, field_name="obligation_id")
        validate_decimal(
            self.evaluated_through_receiver_coordinate,
            field_name="evaluated_through_receiver_coordinate",
        )
        if not self.evaluator.deterministic:
            raise ValueError("prefix obligation evaluator must be deterministic")
        if self.status not in {
            ObligationStatus.SATISFIED,
            ObligationStatus.FAILED,
            ObligationStatus.UNEVALUABLE,
        }:
            raise ValueError("prefix obligation must be satisfied, failed or unevaluable")
        require_sorted_unique_strings(self.evidence_link_ids, field_name="evidence_link_ids")
        _validate_reason_codes(self.reason_codes)
        if self.status is ObligationStatus.SATISFIED:
            if (
                self.first_failure_receiver_coordinate is not None
                or not self.evidence_link_ids
                or self.reason_codes
            ):
                raise ValueError("satisfied prefix obligation requires evidence and no failure")
        elif self.status is ObligationStatus.FAILED:
            if self.first_failure_receiver_coordinate is None:
                raise ValueError("failed prefix obligation requires its first failure coordinate")
            validate_decimal(
                self.first_failure_receiver_coordinate,
                field_name="first_failure_receiver_coordinate",
            )
            if self.first_failure_receiver_coordinate > self.evaluated_through_receiver_coordinate:
                raise ValueError("prefix failure lies after its evaluated interval")
            if not self.evidence_link_ids or not self.reason_codes:
                raise ValueError("failed prefix obligation requires evidence and reasons")
        else:
            if self.first_failure_receiver_coordinate is not None or not self.reason_codes:
                raise ValueError(
                    "unevaluable prefix obligation requires reasons and no imputed failure"
                )


@dataclass(frozen=True, slots=True)
class CausalPrefixCompositionAssessment(CanonicalRecord):
    """Whether one prepared prefix and future action word can be composed."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/causal-prefix-composition-assessment'

    assessment_id: str
    relation: RelationalIdentity
    world_id: str
    prefix_id: str
    action_word_id: str
    causal_support_assessment_id: str
    denominator_id: str
    retained_history_id: str
    receiver_id: str
    horizon_id: str
    prefix_supported: bool
    prefix_cutoff_receiver_coordinate: Decimal
    earliest_action_influence_receiver_coordinate: Decimal | None
    obligations: tuple[PrefixObligationAssessment, ...]
    status: PrefixCompositionStatus
    evidence_links: tuple[EvidenceLink, ...]
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("world_id", self.world_id),
            ("prefix_id", self.prefix_id),
            ("action_word_id", self.action_word_id),
            ("causal_support_assessment_id", self.causal_support_assessment_id),
            ("denominator_id", self.denominator_id),
            ("retained_history_id", self.retained_history_id),
            ("receiver_id", self.receiver_id),
            ("horizon_id", self.horizon_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(
            self.prefix_cutoff_receiver_coordinate,
            field_name="prefix_cutoff_receiver_coordinate",
        )
        if self.earliest_action_influence_receiver_coordinate is not None:
            validate_decimal(
                self.earliest_action_influence_receiver_coordinate,
                field_name="earliest_action_influence_receiver_coordinate",
            )
            if (
                self.earliest_action_influence_receiver_coordinate
                <= self.prefix_cutoff_receiver_coordinate
            ):
                raise ValueError("future action influence must lie strictly after its prefix")
        if self.horizon_id != self.relation.horizon.horizon_id:
            raise ValueError("prefix composition horizon differs from its relation")
        if self.receiver_id not in self.relation.receiver_quantity_ids:
            raise ValueError("prefix composition receiver is outside its relation")
        require_sorted_unique_ids(
            self.obligations,
            attribute="obligation_id",
            field_name="obligations",
        )
        if not self.obligations:
            raise ValueError("prefix composition requires frozen receiver obligations")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        if any(link.world_id != self.world_id for link in self.evidence_links):
            raise ValueError("prefix composition evidence crosses evidence worlds")
        evidence_ids = {link.link_id for link in self.evidence_links}
        if any(
            not set(obligation.evidence_link_ids) <= evidence_ids for obligation in self.obligations
        ):
            raise ValueError("prefix obligation cites evidence outside its composition")
        _validate_reason_codes(self.reason_codes)
        failed_always = tuple(
            obligation
            for obligation in self.obligations
            if obligation.semantics is TemporalPredicateSemantics.ALWAYS_PRESERVED_PATH
            and obligation.status is ObligationStatus.FAILED
            and obligation.first_failure_receiver_coordinate is not None
            and obligation.first_failure_receiver_coordinate
            <= self.prefix_cutoff_receiver_coordinate
        )
        if not self.prefix_supported:
            if self.status is not PrefixCompositionStatus.PREFIX_OUTSIDE_SUPPORT:
                raise ValueError("unsupported prefix requires PREFIX_OUTSIDE_SUPPORT")
        elif self.status is PrefixCompositionStatus.PREFIX_OUTSIDE_SUPPORT:
            raise ValueError("supported prefix cannot be outside support")
        if self.earliest_action_influence_receiver_coordinate is None:
            if self.status not in {
                PrefixCompositionStatus.ACTION_CAUSAL_CONE_MISSING,
                PrefixCompositionStatus.UNEVALUABLE,
            }:
                raise ValueError("missing action cone requires a missing/unevaluable disposition")
        if failed_always:
            if self.status is not PrefixCompositionStatus.OBLIGATION_FAILED_BEFORE_INFLUENCE:
                raise ValueError("prior always-preserved failure forbids defined composition")
        elif self.status is PrefixCompositionStatus.OBLIGATION_FAILED_BEFORE_INFLUENCE:
            raise ValueError(
                "obligation-failure disposition lacks a prior always-preserved failure"
            )
        if self.status is PrefixCompositionStatus.DEFINED:
            if (
                self.earliest_action_influence_receiver_coordinate is None
                or not self.evidence_links
                or self.reason_codes
                or not _retains_categorical_evidence_at(
                    self.evidence_ceiling,
                    EvidenceRung.LOCAL_LAW,
                )
            ):
                raise ValueError("defined prefix composition requires categorical local-law evidence")
            if any(
                obligation.status is ObligationStatus.UNEVALUABLE for obligation in self.obligations
            ):
                raise ValueError("unevaluable obligation cannot yield defined composition")
        elif self.status is PrefixCompositionStatus.UNEVALUABLE:
            if not self.reason_codes:
                raise ValueError("unevaluable prefix composition requires reasons")
        elif not self.evidence_links or not self.reason_codes:
            raise ValueError("opposed prefix composition requires evidence and reasons")
        _validate_visibility(
            evidence_ceiling=self.evidence_ceiling,
            outcome_access=self.outcome_access,
            parent_visibility_ceilings=self.parent_visibility_ceilings,
            visibility_ceiling=self.visibility_ceiling,
            evidence_links=self.evidence_links,
        )


@dataclass(frozen=True, slots=True)
class ResponseAlgebraControlBinding(CanonicalRecord):
    "Exact non-recursive binding from one local law to its causal witnesses."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/response-algebra-control-binding'

    binding_id: str
    law: ObjectIdentity
    relation: ObjectIdentity
    action_word: ObjectIdentity
    causal_support: ObjectIdentity
    prefix_composition: ObjectIdentity
    world_id: str
    denominator_id: str
    retained_history_id: str
    receiver_id: str
    horizon_id: str
    evidence_link_ids: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("world_id", self.world_id),
            ("denominator_id", self.denominator_id),
            ("retained_history_id", self.retained_history_id),
            ("receiver_id", self.receiver_id),
            ("horizon_id", self.horizon_id),
        ):
            validate_stable_id(value, field_name=name)
        expected_schemas = (
            (self.law, ResponseLaw.SCHEMA),
            (self.relation, RelationalIdentity.SCHEMA),
            (self.action_word, LetterActionWord.SCHEMA),
            (self.causal_support, CausalSupportAssessment.SCHEMA),
            (self.prefix_composition, CausalPrefixCompositionAssessment.SCHEMA),
        )
        if any(identity.object_schema != schema for identity, schema in expected_schemas):
            raise ValueError("response-algebra control binding contains a foreign object schema")
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
            allow_empty=False,
        )
        if not self.evidence_ceiling.allows(EvidenceRung.LOCAL_LAW):
            raise ValueError("response-algebra control binding requires local-law evidence")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("non-promotable evidence cannot bind a local-law control law")


def validate_causal_support_against_action_word(
    assessment: CausalSupportAssessment,
    action_word: LetterActionWord,
) -> None:
    """Require the causal witness to describe the exact delivered word."""

    expected = (
        (assessment.action_word_id, action_word.word_id, "action word"),
        (assessment.denominator_id, action_word.denominator_id, "denominator"),
        (assessment.retained_history_id, action_word.retained_history_id, "history"),
        (assessment.receiver_id, action_word.receiver_id, "receiver"),
        (assessment.horizon_id, action_word.horizon_id, "horizon"),
    )
    for observed, actual, label in expected:
        if observed != actual:
            raise ValueError(f"causal support {label} differs from delivered word")
    if assessment.status is CausalSupportStatus.SUPPORTED and not action_word.supported:
        raise ValueError("unsupported delivered word cannot have supported causal evidence")
    if not action_word.letters:
        raise ValueError("causal support assessment requires a non-identity delivered word")
    stage_families = (
        (
            assessment.requested_clock_id,
            assessment.requested_receiver_coordinate,
            tuple(letter.requested for letter in action_word.letters),
            min,
            "requested",
        ),
        (
            assessment.applied_clock_id,
            assessment.applied_receiver_coordinate,
            tuple(letter.applied for letter in action_word.letters),
            min,
            "applied",
        ),
        (
            assessment.realized_clock_id,
            assessment.realized_receiver_coordinate,
            tuple(letter.realized for letter in action_word.letters),
            max,
            "realized",
        ),
    )
    for clock_id, coordinate, stages, reducer, label in stage_families:
        if {stage.clock_id for stage in stages} != {clock_id}:
            raise ValueError(f"causal support {label} clock differs from delivered word")
        if clock_id == assessment.receiver_clock_id:
            expected_coordinate = reducer(stage.clock_coordinate for stage in stages)
            if coordinate != expected_coordinate:
                raise ValueError(f"causal support {label} coordinate differs from delivered word")


def validate_prefix_composition_compatibility(
    composition: CausalPrefixCompositionAssessment,
    causal_support: CausalSupportAssessment,
    action_word: LetterActionWord,
) -> None:
    """Require exact relation, context, causal cone and evidence lineage."""

    validate_causal_support_against_action_word(causal_support, action_word)
    if composition.relation != causal_support.relation:
        raise ValueError("prefix composition and causal support relations differ")
    expected = (
        (composition.world_id, causal_support.world_id, "world"),
        (composition.action_word_id, causal_support.action_word_id, "action word"),
        (
            composition.causal_support_assessment_id,
            causal_support.assessment_id,
            "causal assessment",
        ),
        (composition.denominator_id, causal_support.denominator_id, "denominator"),
        (composition.retained_history_id, causal_support.retained_history_id, "history"),
        (composition.receiver_id, causal_support.receiver_id, "receiver"),
        (composition.horizon_id, causal_support.horizon_id, "horizon"),
    )
    for observed, actual, label in expected:
        if observed != actual:
            raise ValueError(f"prefix composition {label} differs from causal support")
    if (
        composition.earliest_action_influence_receiver_coordinate
        != causal_support.earliest_supported_receiver_coordinate
    ):
        raise ValueError("prefix composition and causal support influence coordinates differ")
    if composition.prefix_cutoff_receiver_coordinate > causal_support.requested_receiver_coordinate:
        raise ValueError("prepared prefix extends beyond the future action request")
    support_link_ids = {link.link_id for link in causal_support.evidence_links}
    composition_link_ids = {link.link_id for link in composition.evidence_links}
    if not support_link_ids <= composition_link_ids:
        raise ValueError("prefix composition omits causal-support evidence lineage")
    if composition.status is PrefixCompositionStatus.DEFINED:
        if causal_support.status is not CausalSupportStatus.SUPPORTED:
            raise ValueError("defined composition requires supported causal evidence")
    elif causal_support.status is CausalSupportStatus.UNEVALUABLE:
        if composition.status is not PrefixCompositionStatus.UNEVALUABLE:
            raise ValueError("unevaluable causal evidence cannot yield evaluated composition")


def build_response_algebra_control_binding(
    *,
    binding_id: str,
    law: ResponseLaw,
    action_word: LetterActionWord,
    causal_support: CausalSupportAssessment,
    prefix_composition: CausalPrefixCompositionAssessment,
) -> ResponseAlgebraControlBinding:
    "Build the exact local-law compatibility object used by later admission authoring."

    validate_prefix_composition_compatibility(
        prefix_composition,
        causal_support,
        action_word,
    )
    if causal_support.status is not CausalSupportStatus.SUPPORTED:
        raise ValueError("Local-law control binding requires supported causal evidence")
    if prefix_composition.status is not PrefixCompositionStatus.DEFINED:
        raise ValueError("Local-law control binding requires defined prefix composition")
    if law.relation != causal_support.relation:
        raise ValueError("response law and causal-support relation differ")
    if law.world_id != causal_support.world_id:
        raise ValueError("response law and causal-support world differ")
    if not law.evidence_ceiling.allows(EvidenceRung.LOCAL_LAW):
        raise ValueError("response law evidence ceiling is below local law")
    if not law.visibility_ceiling.is_promotable:
        raise ValueError("outcome-visible response law cannot bind control")
    required_links = {
        link.link_id: link
        for link in (*causal_support.evidence_links, *prefix_composition.evidence_links)
    }
    law_links = {link.link_id: link for link in law.evidence_links}
    if not set(required_links) <= set(law_links):
        raise ValueError("response law omits causal/prefix evidence links")
    if any(law_links[link_id] != link for link_id, link in required_links.items()):
        raise ValueError("response law reuses an evidence ID with different exact evidence")
    evidence_ceiling = EvidenceCeiling.lowest(
        law.evidence_ceiling,
        causal_support.evidence_ceiling,
        prefix_composition.evidence_ceiling,
    )
    visibility_ceiling = VisibilityCeiling.most_restrictive(
        law.visibility_ceiling,
        causal_support.visibility_ceiling,
        prefix_composition.visibility_ceiling,
    )
    return ResponseAlgebraControlBinding(
        binding_id=binding_id,
        law=ObjectIdentity.from_record(law.law_id, law),
        relation=ObjectIdentity.from_record(law.relation.relation_id, law.relation),
        action_word=ObjectIdentity.from_record(action_word.word_id, action_word),
        causal_support=ObjectIdentity.from_record(causal_support.assessment_id, causal_support),
        prefix_composition=ObjectIdentity.from_record(
            prefix_composition.assessment_id,
            prefix_composition,
        ),
        world_id=law.world_id,
        denominator_id=causal_support.denominator_id,
        retained_history_id=causal_support.retained_history_id,
        receiver_id=causal_support.receiver_id,
        horizon_id=causal_support.horizon_id,
        evidence_link_ids=tuple(sorted(required_links)),
        evidence_ceiling=evidence_ceiling,
        visibility_ceiling=visibility_ceiling,
    )


def response_algebra_control_extension(
    binding: ResponseAlgebraControlBinding,
) -> ExtensionBinding:
    "Return the namespaced immutable reference used by additive admission records."

    return ExtensionBinding(
        namespace="response-algebra-control",
        schema=binding.SCHEMA,
        payload_sha256=binding.fingerprint(),
    )


def decode_causal_support_assessment(payload: bytes) -> CausalSupportAssessment:
    return decode_canonical_bytes(
        payload,
        CausalSupportAssessment,
        maximum_bytes=MAX_RESPONSE_ALGEBRA_CONTROL_RECORD_BYTES,
    )


def decode_causal_prefix_composition_assessment(
    payload: bytes,
) -> CausalPrefixCompositionAssessment:
    return decode_canonical_bytes(
        payload,
        CausalPrefixCompositionAssessment,
        maximum_bytes=MAX_RESPONSE_ALGEBRA_CONTROL_RECORD_BYTES,
    )


def decode_response_algebra_control_binding(
    payload: bytes,
) -> ResponseAlgebraControlBinding:
    return decode_canonical_bytes(
        payload,
        ResponseAlgebraControlBinding,
        maximum_bytes=MAX_RESPONSE_ALGEBRA_CONTROL_RECORD_BYTES,
    )
