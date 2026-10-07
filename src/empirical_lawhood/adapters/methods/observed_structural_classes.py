"Complete, archive-aware and label-blind observed structural-class assessment.\n\nThe module is adapter-local.  It does not modify the frozen kernel structural\nclass prototype and it cannot emit a structural-class promotion.  Cohomology is\ncarried as an orthogonal disposition rather than used as an ISDG axis.\n"

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from enum import StrEnum
import json
from typing import ClassVar, Mapping, cast

from empirical_lawhood.adapters.methods.structural_response_classes import (
    ISDG_AXIS_IDS,
    ISDG_SIGNATURE_ID,
    AxisDisposition,
    StructuralMechanismDisposition,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_document_shape,
    validate_nonempty,
    validate_stable_id,
)


class EvidenceWorld(StrEnum):
    TRUTH_KNOWN_GENERATED = "TRUTH_KNOWN_GENERATED"
    RESETTABLE_SIMULATOR = "RESETTABLE_SIMULATOR"
    RETROSPECTIVE_PHYSICAL_ARCHIVE = "RETROSPECTIVE_PHYSICAL_ARCHIVE"
    PROSPECTIVE_PHYSICAL = "PROSPECTIVE_PHYSICAL"


class AblationSemantics(StrEnum):
    SUBSTRATE_INTERVENTION = "SUBSTRATE_INTERVENTION"
    SIMULATED_STATE_INTERVENTION = "SIMULATED_STATE_INTERVENTION"
    RECEIVER_CHANNEL_WITHHOLDING = "RECEIVER_CHANNEL_WITHHOLDING"
    NOT_PERFORMED = "NOT_PERFORMED"


class RestorationSemantics(StrEnum):
    PHYSICAL_RESTORATION = "PHYSICAL_RESTORATION"
    SIMULATED_STATE_RESTORATION = "SIMULATED_STATE_RESTORATION"
    PREDICTIVE_CHANNEL_REINTRODUCTION = "PREDICTIVE_CHANNEL_REINTRODUCTION"
    NOT_PERFORMED = "NOT_PERFORMED"


@dataclass(frozen=True, slots=True)
class InterfaceSubsetScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/interface-subset-score'

    subset_id: str
    coordinate_ids: tuple[str, ...]
    heldout_development_error: Decimal
    eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.subset_id, field_name="subset_id")
        require_sorted_unique_strings(
            self.coordinate_ids,
            field_name="coordinate_ids",
            allow_empty=True,
        )
        validate_decimal(
            self.heldout_development_error,
            field_name="heldout_development_error",
            minimum=Decimal("0"),
        )
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=True,
        )
        if not self.eligible and not self.reason_codes:
            raise ValueError("ineligible subset requires a reason")


@dataclass(frozen=True, slots=True)
class DevelopmentSubsetSelection(CanonicalRecord):
    """Frozen, development-only evidence for the smallest sufficient subset."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/development-subset-selection'

    selection_id: str
    development_partition_id: str
    candidate_coordinate_ids: tuple[str, ...]
    causally_available_coordinate_ids: tuple[str, ...]
    scores: tuple[InterfaceSubsetScore, ...]
    selected_subset_id: str
    sufficient_error_allowance: Decimal
    parsimony_margin: Decimal
    independent_development_unit_count: int
    evaluation_outcomes_accessed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.selection_id, field_name="selection_id")
        validate_stable_id(
            self.development_partition_id,
            field_name="development_partition_id",
        )
        require_sorted_unique_strings(
            self.candidate_coordinate_ids,
            field_name="candidate_coordinate_ids",
        )
        require_sorted_unique_strings(
            self.causally_available_coordinate_ids,
            field_name="causally_available_coordinate_ids",
        )
        if not set(self.causally_available_coordinate_ids).issubset(self.candidate_coordinate_ids):
            raise ValueError("causally available coordinates must be candidates")
        require_sorted_unique_ids(self.scores, attribute="subset_id", field_name="scores")
        validate_stable_id(self.selected_subset_id, field_name="selected_subset_id")
        validate_decimal(
            self.sufficient_error_allowance,
            field_name="sufficient_error_allowance",
            minimum=Decimal("0"),
        )
        validate_decimal(
            self.parsimony_margin,
            field_name="parsimony_margin",
            minimum=Decimal("0"),
        )
        if self.independent_development_unit_count < 1:
            raise ValueError("development subset selection requires independent units")
        if self.evaluation_outcomes_accessed:
            raise ValueError("evaluation-visible subset selection is invalid")
        selected = next(
            (score for score in self.scores if score.subset_id == self.selected_subset_id),
            None,
        )
        if selected is None or not selected.eligible:
            raise ValueError("selected subset is absent or ineligible")
        selected_coordinates = set(selected.coordinate_ids)
        if not selected_coordinates.issubset(self.causally_available_coordinate_ids):
            raise ValueError("selected subset contains post-cutoff or unavailable state")
        if selected.heldout_development_error > self.sufficient_error_allowance:
            raise ValueError("selected subset is not sufficient on development")
        for score in self.scores:
            coordinates = set(score.coordinate_ids)
            if not coordinates.issubset(self.candidate_coordinate_ids):
                raise ValueError("subset score contains a noncandidate coordinate")
            if not score.eligible:
                continue
            if coordinates < selected_coordinates:
                if score.heldout_development_error <= self.sufficient_error_allowance:
                    raise ValueError("selected subset is not the smallest sufficient subset")
            if coordinates > selected_coordinates:
                improvement = selected.heldout_development_error - score.heldout_development_error
                if improvement > self.parsimony_margin:
                    raise ValueError("larger subset improves beyond the parsimony margin")

    @property
    def selected_score(self) -> InterfaceSubsetScore:
        return next(score for score in self.scores if score.subset_id == self.selected_subset_id)


@dataclass(frozen=True, slots=True)
class ObservedStructuralClassAxisEvidence(CanonicalRecord):
    """One truth-blind observed axis input; ``supported=None`` means missing."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/observed-structural-class-axis-evidence'

    axis_id: str
    supported: bool | None
    estimate: Decimal | None
    threshold: Decimal | None
    independent_unit_count: int
    preparation_ids: tuple[str, ...]
    lot_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.axis_id not in ISDG_AXIS_IDS:
            raise ValueError("unknown ISDG axis")
        if self.estimate is not None:
            validate_decimal(self.estimate, field_name="estimate")
        if self.threshold is not None:
            validate_decimal(self.threshold, field_name="threshold", minimum=Decimal("0"))
        if self.independent_unit_count < 0:
            raise ValueError("independent unit count must be nonnegative")
        require_sorted_unique_strings(
            self.preparation_ids,
            field_name="preparation_ids",
            allow_empty=True,
        )
        require_sorted_unique_strings(self.lot_ids, field_name="lot_ids", allow_empty=True)
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=True,
        )
        if self.supported is not True and not self.reason_codes:
            raise ValueError("nonpassing or missing axis evidence requires a reason")


@dataclass(frozen=True, slots=True)
class ObservedStructuralClassObservedEvidence(CanonicalRecord):
    """Complete label-free evaluator input."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/observed-structural-class-observed-evidence'

    evidence_id: str
    context_id: str
    evidence_world: EvidenceWorld
    development_selection: DevelopmentSubsetSelection
    axes: tuple[ObservedStructuralClassAxisEvidence, ...]
    ablation_semantics: AblationSemantics
    restoration_semantics: RestorationSemantics
    source_backed_action: bool
    causal_localization_supported: bool
    lot_identity_available: bool
    cohomology_disposition: str

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        validate_stable_id(self.context_id, field_name="context_id")
        require_sorted_unique_ids(self.axes, attribute="axis_id", field_name="axes")
        if tuple(axis.axis_id for axis in self.axes) != ISDG_AXIS_IDS:
            raise ValueError("ISDG observed evidence requires exactly eight ordered axes")
        validate_nonempty(
            self.cohomology_disposition,
            field_name="cohomology_disposition",
        )


@dataclass(frozen=True, slots=True)
class ObservedStructuralClassAxisResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/observed-structural-class-axis-result'

    axis_id: str
    disposition: AxisDisposition
    estimate: Decimal | None
    threshold: Decimal | None
    independent_unit_count: int
    preparation_ids: tuple[str, ...]
    lot_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.axis_id not in ISDG_AXIS_IDS:
            raise ValueError("unknown ISDG axis")
        if self.estimate is not None:
            validate_decimal(self.estimate, field_name="estimate")
        if self.threshold is not None:
            validate_decimal(self.threshold, field_name="threshold", minimum=Decimal("0"))
        if self.independent_unit_count < 0:
            raise ValueError("independent unit count must be nonnegative")
        require_sorted_unique_strings(
            self.preparation_ids,
            field_name="preparation_ids",
            allow_empty=True,
        )
        require_sorted_unique_strings(self.lot_ids, field_name="lot_ids", allow_empty=True)
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=True,
        )
        if self.disposition is not AxisDisposition.PASS and not self.reason_codes:
            raise ValueError("nonpassing axis requires a reason")


@dataclass(frozen=True, slots=True)
class ObservedStructuralClassAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/observed-structural-class-assessment'

    assessment_id: str
    context_id: str
    signature_id: str
    evidence_world: EvidenceWorld
    development_selection_id: str
    axis_results: tuple[ObservedStructuralClassAxisResult, ...]
    mechanism_disposition: StructuralMechanismDisposition
    cohomology_disposition: str
    structural_class_outcome: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.context_id, field_name="context_id")
        validate_stable_id(
            self.development_selection_id,
            field_name="development_selection_id",
        )
        if self.signature_id != ISDG_SIGNATURE_ID:
            raise ValueError("ISDG signature differs")
        require_sorted_unique_ids(
            self.axis_results,
            attribute="axis_id",
            field_name="axis_results",
        )
        if tuple(axis.axis_id for axis in self.axis_results) != ISDG_AXIS_IDS:
            raise ValueError("ISDG assessment requires exactly eight ordered axes")
        validate_nonempty(
            self.cohomology_disposition,
            field_name="cohomology_disposition",
        )
        if self.structural_class_outcome != "STRUCTURAL_CLASS_NONATTEMPT_ARCHIVAL_ONLY":
            raise ValueError("Observed structural-class assessment cannot emit a structural-class promotion")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        all_pass = all(result.disposition is AxisDisposition.PASS for result in self.axis_results)
        if (
            self.mechanism_disposition
            is StructuralMechanismDisposition.INTERFACE_STATE_DEPENDENT_GLOBALIZATION
        ) != all_pass:
            raise ValueError("ISDG disposition and eight-axis conjunction disagree")


def _axis_result(
    evidence: ObservedStructuralClassAxisEvidence,
    *,
    forced_disposition: AxisDisposition | None = None,
    forced_reason: str | None = None,
) -> ObservedStructuralClassAxisResult:
    disposition = forced_disposition
    reasons = set(evidence.reason_codes)
    if disposition is None:
        if evidence.supported is True:
            disposition = AxisDisposition.PASS
        elif evidence.supported is False:
            disposition = AxisDisposition.FAIL
        else:
            disposition = AxisDisposition.UNEVALUABLE
    if forced_reason is not None:
        reasons.add(forced_reason)
    return ObservedStructuralClassAxisResult(
        axis_id=evidence.axis_id,
        disposition=disposition,
        estimate=evidence.estimate,
        threshold=evidence.threshold,
        independent_unit_count=evidence.independent_unit_count,
        preparation_ids=evidence.preparation_ids,
        lot_ids=evidence.lot_ids,
        reason_codes=tuple(sorted(reasons)),
    )


def evaluate_isdg(evidence: ObservedStructuralClassObservedEvidence) -> ObservedStructuralClassAssessment:
    """Evaluate only declared observations; no mechanism label is accepted."""

    results: list[ObservedStructuralClassAxisResult] = []
    archive = evidence.evidence_world is EvidenceWorld.RETROSPECTIVE_PHYSICAL_ARCHIVE
    for axis in evidence.axes:
        forced: AxisDisposition | None = None
        reason: str | None = None
        if axis.axis_id == ISDG_AXIS_IDS[0] and archive and not evidence.source_backed_action:
            forced = AxisDisposition.UNEVALUABLE
            reason = "ARCHIVE_DELIVERED_ACTION_UNOBSERVED"
        elif axis.axis_id == ISDG_AXIS_IDS[2]:
            if evidence.development_selection.selected_score.heldout_development_error > (
                evidence.development_selection.sufficient_error_allowance
            ):
                forced = AxisDisposition.FAIL
                reason = "DEVELOPMENT_SELECTED_SUBSET_NOT_SUFFICIENT"
        elif axis.axis_id == ISDG_AXIS_IDS[3]:
            if (
                archive
                and evidence.ablation_semantics is AblationSemantics.RECEIVER_CHANNEL_WITHHOLDING
            ):
                forced = AxisDisposition.UNEVALUABLE
                reason = "ARCHIVE_CHANNEL_WITHHOLDING_NOT_SUBSTRATE_ABLATION"
            elif evidence.ablation_semantics is AblationSemantics.NOT_PERFORMED:
                forced = AxisDisposition.UNEVALUABLE
                reason = "ABLATION_NOT_PERFORMED"
        elif axis.axis_id == ISDG_AXIS_IDS[4] and not evidence.causal_localization_supported:
            forced = AxisDisposition.UNEVALUABLE
            reason = "CAUSAL_LOCALIZATION_NOT_SUPPORTED"
        elif axis.axis_id == ISDG_AXIS_IDS[5]:
            if (
                archive
                and evidence.restoration_semantics
                is RestorationSemantics.PREDICTIVE_CHANNEL_REINTRODUCTION
            ):
                forced = AxisDisposition.UNEVALUABLE
                reason = "PREDICTIVE_REINTRODUCTION_NOT_PHYSICAL_RESCUE"
            elif evidence.restoration_semantics is RestorationSemantics.NOT_PERFORMED:
                forced = AxisDisposition.UNEVALUABLE
                reason = "RESTORATION_NOT_PERFORMED"
        elif axis.axis_id == ISDG_AXIS_IDS[7] and archive and not evidence.lot_identity_available:
            forced = AxisDisposition.UNEVALUABLE
            reason = "ARCHIVE_LOT_IDENTITY_UNAVAILABLE"
        results.append(_axis_result(axis, forced_disposition=forced, forced_reason=reason))

    dispositions = tuple(result.disposition for result in results)
    if any(value is AxisDisposition.INVALID for value in dispositions):
        mechanism = StructuralMechanismDisposition.INVALID
        mechanism_reason = "AT_LEAST_ONE_AXIS_INVALID"
    elif any(value is AxisDisposition.UNEVALUABLE for value in dispositions):
        mechanism = StructuralMechanismDisposition.UNEVALUABLE
        mechanism_reason = "AT_LEAST_ONE_AXIS_UNEVALUABLE"
    elif all(value is AxisDisposition.PASS for value in dispositions):
        mechanism = StructuralMechanismDisposition.INTERFACE_STATE_DEPENDENT_GLOBALIZATION
        mechanism_reason = "ALL_EIGHT_ISDG_AXES_PASS"
    elif results[6].disposition is AxisDisposition.FAIL:
        mechanism = StructuralMechanismDisposition.RECEIVER_COBOUNDARY_ONLY
        mechanism_reason = "COUNTERFEIT_REJECTION_FAILED"
    elif results[3].disposition is AxisDisposition.FAIL:
        mechanism = StructuralMechanismDisposition.COMPATIBLE_GLOBALIZATION
        mechanism_reason = "ABLATION_EXCESS_FAILURE_OPPOSED"
    elif results[5].disposition is AxisDisposition.FAIL:
        mechanism = StructuralMechanismDisposition.PERSISTENT_OBSTRUCTION_CONTRAST
        mechanism_reason = "RESTORATION_RESCUE_OPPOSED"
    else:
        mechanism = StructuralMechanismDisposition.APPARENT_ANALOGY_ONLY
        mechanism_reason = "NONCOMPENSATING_CONJUNCTION_OPPOSED"

    return ObservedStructuralClassAssessment(
        assessment_id=f"isdg.{evidence.context_id}",
        context_id=evidence.context_id,
        signature_id=ISDG_SIGNATURE_ID,
        evidence_world=evidence.evidence_world,
        development_selection_id=evidence.development_selection.selection_id,
        axis_results=tuple(results),
        mechanism_disposition=mechanism,
        cohomology_disposition=evidence.cohomology_disposition,
        structural_class_outcome="STRUCTURAL_CLASS_NONATTEMPT_ARCHIVAL_ONLY",
        reason_codes=(mechanism_reason,),
    )


def _decimal_from_document(value: object) -> Decimal:
    if not isinstance(value, Mapping) or set(value) != {"decimal"}:
        raise ValueError("canonical decimal envelope differs")
    raw = value["decimal"]
    if not isinstance(raw, str):
        raise ValueError("canonical decimal text differs")
    result = Decimal(raw)
    validate_decimal(result, field_name="canonical_decimal")
    return result


def decode_development_subset_selection(
    payload: bytes,
) -> DevelopmentSubsetSelection:
    """Strict decoder for the leakage-sensitive development selection."""

    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("development-subset payload is not JSON") from error
    if not isinstance(document, Mapping):
        raise ValueError("development-subset document is not a mapping")
    raw = validate_document_shape(
        cast(Mapping[str, object], document),
        expected_schema=DevelopmentSubsetSelection.SCHEMA,
        expected_version=DevelopmentSubsetSelection.VERSION,
        field_names=frozenset(field.name for field in fields(DevelopmentSubsetSelection)),
    )

    def string_tuple(name: str) -> tuple[str, ...]:
        value = raw[name]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{name} differs")
        return tuple(value)

    scores_raw = raw["scores"]
    if not isinstance(scores_raw, list):
        raise ValueError("development subset scores differ")
    scores: list[InterfaceSubsetScore] = []
    expected_score_fields = {field.name for field in fields(InterfaceSubsetScore)}
    for value in scores_raw:
        if not isinstance(value, Mapping):
            raise ValueError("development subset score is not a mapping")
        score_document = cast(Mapping[str, object], value)
        if set(score_document) != {"schema", "value", "version"}:
            raise ValueError("development subset score envelope differs")
        if (
            score_document["schema"] != InterfaceSubsetScore.SCHEMA
            or score_document["version"] != InterfaceSubsetScore.VERSION
        ):
            raise ValueError("development subset score identity differs")
        score_value = score_document["value"]
        if not isinstance(score_value, Mapping) or set(score_value) != expected_score_fields:
            raise ValueError("development subset score fields differ")
        coordinates = score_value["coordinate_ids"]
        reasons = score_value["reason_codes"]
        if not isinstance(coordinates, list) or any(
            not isinstance(item, str) for item in coordinates
        ):
            raise ValueError("development subset coordinate ids differ")
        if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
            raise ValueError("development subset reason codes differ")
        scores.append(
            InterfaceSubsetScore(
                subset_id=str(score_value["subset_id"]),
                coordinate_ids=tuple(coordinates),
                heldout_development_error=_decimal_from_document(
                    score_value["heldout_development_error"]
                ),
                eligible=bool(score_value["eligible"]),
                reason_codes=tuple(reasons),
            )
        )
    independent_unit_count = raw["independent_development_unit_count"]
    if not isinstance(independent_unit_count, int) or isinstance(independent_unit_count, bool):
        raise ValueError("independent development unit count differs")
    return DevelopmentSubsetSelection(
        selection_id=str(raw["selection_id"]),
        development_partition_id=str(raw["development_partition_id"]),
        candidate_coordinate_ids=string_tuple("candidate_coordinate_ids"),
        causally_available_coordinate_ids=string_tuple("causally_available_coordinate_ids"),
        scores=tuple(scores),
        selected_subset_id=str(raw["selected_subset_id"]),
        sufficient_error_allowance=_decimal_from_document(raw["sufficient_error_allowance"]),
        parsimony_margin=_decimal_from_document(raw["parsimony_margin"]),
        independent_development_unit_count=independent_unit_count,
        evaluation_outcomes_accessed=bool(raw["evaluation_outcomes_accessed"]),
    )


__all__ = [
    'AblationSemantics',
    'DevelopmentSubsetSelection',
    'EvidenceWorld',
    'ObservedStructuralClassAssessment',
    'ObservedStructuralClassAxisEvidence',
    'ObservedStructuralClassAxisResult',
    'ObservedStructuralClassObservedEvidence',
    'InterfaceSubsetScore',
    'RestorationSemantics',
    'decode_development_subset_selection',
    'evaluate_isdg',
]
