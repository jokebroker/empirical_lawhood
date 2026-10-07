"Finite, simultaneous-set admission operands beside controlled-map reachability.\n\nThese records carry qualified absolute predictions, never controllability rank.\nContainment is a deterministic projection; custody and qualification remain the\nexisting production owners' responsibility.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import ActionOccurrence, OccurrenceActionWord
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import AvailabilitySpec, CausalPhase, HorizonSpec, InformationCutoff
from empirical_lawhood.planning.evidence_geometry import LawEvaluationBindingDisposition, LawMemberEvaluationBinding, ReceiptAdmissionPlannedCoordinate, validate_law_evaluation_binding, _validate_admission_receipt_evidence


class FiniteReadoutKind(StrEnum):
    ENDPOINT = "ENDPOINT"
    EARLY = "EARLY"
    LATE = "LATE"
    WINDOW_MAXIMUM = "WINDOW_MAXIMUM"


@dataclass(frozen=True, slots=True)
class FiniteResponseCoordinate(CanonicalRecord):
    """A qualified native receiver readout, including its observation operator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-response-coordinate'

    coordinate_id: str
    quantity_id: str
    receiver_id: str
    readout: ObjectIdentity
    kind: FiniteReadoutKind
    unit: str
    frame_id: str
    clock_id: str
    start: Decimal
    end: Decimal

    def __post_init__(self) -> None:
        for name in ("coordinate_id", "quantity_id", "receiver_id", "frame_id", "clock_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_nonempty(self.unit, field_name="unit")
        validate_decimal(self.start, field_name="start")
        validate_decimal(self.end, field_name="end")
        if not isinstance(self.kind, FiniteReadoutKind) or self.start > self.end:
            raise ValueError("finite readout requires its kind and ordered clock interval")


@dataclass(frozen=True, slots=True)
class FiniteResponseBound(CanonicalRecord):
    """One component of a jointly calibrated box, in the declared target frame.

    Baseline + learned response + causal frame offset equals the absolute mean.
    Bounds already contain that baseline and offset. Numerical uncertainty is
    an additional nonnegative outward expansion, never silently subtracted.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-response-bound'

    bound_id: str
    qualification_view_id: str
    coordinate: FiniteResponseCoordinate
    absolute_baseline: Decimal
    response_delta: Decimal
    handoff_frame_offset: Decimal
    lower: Decimal
    upper: Decimal
    numerical_floor: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.bound_id, field_name="bound_id")
        validate_stable_id(self.qualification_view_id, field_name="qualification_view_id")
        for name in (
            "absolute_baseline",
            "response_delta",
            "handoff_frame_offset",
            "lower",
            "upper",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        validate_decimal(self.numerical_floor, field_name="numerical_floor", minimum=Decimal(0))
        if not self.lower <= self.absolute_mean <= self.upper:
            raise ValueError("finite bounds must contain the absolute baseline-plus-response mean")

    @property
    def absolute_mean(self) -> Decimal:
        return self.absolute_baseline + self.response_delta + self.handoff_frame_offset

    @property
    def expanded_interval(self) -> tuple[Decimal, Decimal]:
        return self.lower - self.numerical_floor, self.upper + self.numerical_floor


@dataclass(frozen=True, slots=True)
class FiniteBinary64ResponseBound(FiniteResponseBound):
    """Round-trip endpoints and original halfwidth, without double numerical expansion."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-binary64-response-bound'
    calibrated_halfwidth: Decimal

    def __post_init__(self) -> None:
        FiniteResponseBound.__post_init__(self)
        validate_decimal(
            self.calibrated_halfwidth, field_name="calibrated_halfwidth", minimum=Decimal(0)
        )
        mean, width = float(self.absolute_mean), float(self.calibrated_halfwidth)
        if self.numerical_floor != 0 or (self.lower, self.upper) != (
            Decimal(repr(mean - width)),
            Decimal(repr(mean + width)),
        ):
            raise ValueError("binary64 response endpoints changed arithmetic or added eta twice")


class FiniteSetDisposition(StrEnum):
    AVAILABLE = "AVAILABLE"
    OUTSIDE_SUPPORT = "OUTSIDE_SUPPORT"
    UNAVAILABLE = "UNAVAILABLE"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class FiniteJointCalibration(CanonicalRecord):
    """One simultaneous box calibration across the full declared action/readout/view vector."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-joint-calibration'

    calibration_id: str
    receipt: ObjectIdentity
    population: ObjectIdentity
    action_words: tuple[ObjectIdentity, ...]
    coordinates: tuple[FiniteResponseCoordinate, ...]
    qualification_view_ids: tuple[str, ...]
    coverage_probability: Decimal
    calibration_artifact: ArtifactIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.calibration_id, field_name="calibration_id")
        require_sorted_unique_ids(
            self.action_words, attribute="object_id", field_name="action_words"
        )
        require_sorted_unique_ids(
            self.coordinates, attribute="coordinate_id", field_name="coordinates"
        )
        require_sorted_unique_strings(
            self.qualification_view_ids,
            field_name="qualification_view_ids",
            allow_empty=False,
        )
        if not 1 <= len(self.action_words) <= 32 or not 1 <= len(self.coordinates) <= 32:
            raise ValueError("joint calibration exceeds the bounded action/readout roster")
        if len(self.qualification_view_ids) > 32:
            raise ValueError("joint calibration exceeds the bounded numerical-view roster")
        if not self.action_words or not self.coordinates:
            raise ValueError("joint calibration requires its complete action/readout roster")
        if any(word.object_schema != OccurrenceActionWord.SCHEMA for word in self.action_words):
            raise ValueError("joint calibration requires exact native action words")
        validate_decimal(self.coverage_probability, field_name="coverage_probability")
        if not Decimal(0) < self.coverage_probability < Decimal(1):
            raise ValueError("joint calibration coverage must be strictly between zero and one")


@dataclass(frozen=True, slots=True)
class FinitePolicyPathCalibration(FiniteJointCalibration):
    "A frozen consumer's chosen-path calibration, with a complete query menu.\n\n    Structural compatibility with the finite box machinery does not promote\n    coverage to unselected actions or alternative consumers. The complete menu\n    records queries; empirical coverage belongs only to the frozen policy path.\n    It cannot be decoded as a FiniteJointCalibration record.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-policy-path-calibration'
    consumer_policy: ObjectIdentity

    def __post_init__(self) -> None:
        FiniteJointCalibration.__post_init__(self)
        if self.consumer_policy.object_schema != 'empirical-lawhood/planning/frozen-feedback-consumer':
            raise ValueError("policy-path calibration requires an exact frozen consumer")


@dataclass(frozen=True, slots=True)
class FiniteResponseSet(CanonicalRecord):
    """One planned word's complete per-view slice of a simultaneous response box."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-response-set'

    set_id: str
    planned_coordinate: ReceiptAdmissionPlannedCoordinate
    law_payload: ArtifactIdentity
    evaluation_bindings: tuple[LawMemberEvaluationBinding, ...]
    joint_calibration: FiniteJointCalibration | FinitePolicyPathCalibration
    public_handoff: ObjectIdentity
    frame_translation: ObjectIdentity
    handoff_availability: AvailabilitySpec
    information_cutoff: InformationCutoff
    outcome_access: OutcomeAccess
    support_assessment: ObjectIdentity
    expiry_assessment: ObjectIdentity
    effort_assessment: ObjectIdentity
    preservation_assessment: ObjectIdentity
    disposition: FiniteSetDisposition
    bounds: tuple[FiniteResponseBound | FiniteBinary64ResponseBound, ...]
    source_artifacts: tuple[ArtifactIdentity, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.set_id, field_name="set_id")
        if type(self) is FiniteResponseSet and isinstance(
            self.joint_calibration, FinitePolicyPathCalibration
        ):
            raise ValueError("policy-path calibration cannot masquerade as a simultaneous box")
        require_sorted_unique_ids(
            self.evaluation_bindings,
            attribute="qualification_view_id",
            field_name="evaluation_bindings",
        )
        views = self.planned_coordinate.qualification_view_ids
        if tuple(b.qualification_view_id for b in self.evaluation_bindings) != views:
            raise ValueError("finite set must bind every planned qualification view exactly once")
        for binding in self.evaluation_bindings:
            validate_law_evaluation_binding(self.planned_coordinate, binding)
        words = {binding.action_word for binding in self.evaluation_bindings}
        if len(words) != 1 or not words.issubset(set(self.joint_calibration.action_words)):
            raise ValueError("finite set action differs across views or from joint calibration")
        if views != self.joint_calibration.qualification_view_ids:
            raise ValueError("finite set and joint calibration qualification rosters differ")
        if not self.information_cutoff.phase.precedes_or_equals(CausalPhase.PRE_ACTION):
            raise ValueError("finite admission prediction cutoff must precede the task action")
        self.information_cutoff.require_allows(self.handoff_availability)
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("finite admission sets cannot consume task outcomes")
        require_sorted_unique_ids(self.bounds, attribute="bound_id", field_name="bounds")
        require_sorted_unique_ids(
            self.source_artifacts,
            attribute="artifact_id",
            field_name="source_artifacts",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not isinstance(self.disposition, FiniteSetDisposition):
            raise ValueError("finite set requires a typed disposition")
        if self.disposition is not FiniteSetDisposition.AVAILABLE:
            if self.bounds or not self.reason_codes:
                raise ValueError("unavailable finite sets require reasons and cannot impute bounds")
            return
        if self.reason_codes or not self.source_artifacts:
            raise ValueError("available finite sets require artifacts and no missingness reasons")
        if any(
            b.evaluation_disposition is not LawEvaluationBindingDisposition.SUPPORTED
            for b in self.evaluation_bindings
        ):
            raise ValueError("available finite sets require every qualified supported evaluation")
        expected = {(view, c) for view in views for c in self.joint_calibration.coordinates}
        actual = {(b.qualification_view_id, b.coordinate) for b in self.bounds}
        if actual != expected or len(self.bounds) != len(expected):
            raise ValueError("finite set must retain the complete joint readout/view product")


@dataclass(frozen=True, slots=True)
class FinitePolicyPathResponseSet(FiniteResponseSet):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-policy-path-response-set'
    joint_calibration: FinitePolicyPathCalibration

    def __post_init__(self) -> None:
        FiniteResponseSet.__post_init__(self)
        if not isinstance(self.joint_calibration, FinitePolicyPathCalibration):
            raise ValueError("policy-path box requires its explicit narrower calibration scope")


@dataclass(frozen=True, slots=True)
class FiniteTargetInterval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-target-interval'

    coordinate: FiniteResponseCoordinate
    lower: Decimal
    upper: Decimal

    def __post_init__(self) -> None:
        validate_decimal(self.lower, field_name="lower")
        validate_decimal(self.upper, field_name="upper")
        if self.lower > self.upper:
            raise ValueError("target interval bounds are reversed")


@dataclass(frozen=True, slots=True)
class FiniteTargetBox(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-target-box'

    box_id: str
    intervals: tuple[FiniteTargetInterval, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.box_id, field_name="box_id")
        ids = tuple(i.coordinate.coordinate_id for i in self.intervals)
        require_sorted_unique_strings(ids, field_name="target coordinates", allow_empty=False)


class FiniteTaskFunctionalKind(StrEnum):
    ENDPOINT_BOX = "ENDPOINT_BOX"
    DISJOINT_ENDPOINT_UNION = "DISJOINT_ENDPOINT_UNION"
    EARLY_LATE_BOX = "EARLY_LATE_BOX"
    WINDOW_MAXIMUM_BOX = "WINDOW_MAXIMUM_BOX"


@dataclass(frozen=True, slots=True)
class FiniteTaskFunctionalSpec(CanonicalRecord):
    """Closed installed functionals; every interval face is conjunctive.

    Union components must be strictly disjoint closed boxes. A connected
    prediction box is contained in this union iff contained in one component.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-task-functional-spec'

    functional_id: str
    kind: FiniteTaskFunctionalKind
    predeclared_family: ObjectIdentity
    target_reveal: ObjectIdentity
    target_availability: AvailabilitySpec
    boxes: tuple[FiniteTargetBox, ...]
    tie_order: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.functional_id, field_name="functional_id")
        require_sorted_unique_ids(self.boxes, attribute="box_id", field_name="boxes")
        if len(self.boxes) > 32:
            raise ValueError("finite functional exceeds its bounded component count")
        if not self.boxes or not isinstance(self.kind, FiniteTaskFunctionalKind):
            raise ValueError("finite functional requires typed nonempty target boxes")
        axes = tuple(i.coordinate for i in self.boxes[0].intervals)
        if any(tuple(i.coordinate for i in b.intervals) != axes for b in self.boxes):
            raise ValueError("union components must use identical native coordinate contracts")
        if (
            self.kind is not FiniteTaskFunctionalKind.DISJOINT_ENDPOINT_UNION
            and len(self.boxes) != 1
        ):
            raise ValueError("box functional requires exactly one component")
        if self.kind is FiniteTaskFunctionalKind.EARLY_LATE_BOX:
            if {c.kind for c in axes} != {FiniteReadoutKind.EARLY, FiniteReadoutKind.LATE}:
                raise ValueError("early/late functional requires both qualified readout kinds")
        elif self.kind is FiniteTaskFunctionalKind.WINDOW_MAXIMUM_BOX:
            if not any(c.kind is FiniteReadoutKind.WINDOW_MAXIMUM for c in axes) or any(
                c.kind not in (FiniteReadoutKind.WINDOW_MAXIMUM, FiniteReadoutKind.ENDPOINT)
                for c in axes
            ):
                raise ValueError("window-maximum functional requires its qualified maximum readout")
        elif any(c.kind is not FiniteReadoutKind.ENDPOINT for c in axes):
            raise ValueError("endpoint functional cannot substitute a path readout")
        for index, left in enumerate(self.boxes):
            for right in self.boxes[index + 1 :]:
                if not any(
                    a.upper < b.lower or b.upper < a.lower
                    for a, b in zip(left.intervals, right.intervals, strict=True)
                ):
                    raise ValueError("endpoint union requires strictly disjoint closed boxes")
        if len(self.tie_order) != len(self.boxes) or set(self.tie_order) != {
            b.box_id for b in self.boxes
        }:
            raise ValueError("functional tie order must cover every component exactly once")


@dataclass(frozen=True, slots=True)
class FiniteResponseSetReachabilityRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-response-set-reachability-request'

    request_id: str
    response_set: FiniteResponseSet | FinitePolicyPathResponseSet
    task: FiniteTaskFunctionalSpec
    action_word: OccurrenceActionWord
    action_occurrences: tuple[ObjectIdentity, ...]
    information_cutoff: InformationCutoff

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")
        self.information_cutoff.require_allows(self.task.target_availability)
        if self.information_cutoff != self.response_set.information_cutoff:
            raise ValueError("finite reachability and prediction use different causal cutoffs")
        word_identity = ObjectIdentity.from_record(self.action_word.word_id, self.action_word)
        if any(b.action_word != word_identity for b in self.response_set.evaluation_bindings):
            raise ValueError("finite reachability uses another exact action word")
        expected_occurrences = tuple(
            ObjectIdentity.from_record(o.occurrence_id, o) for o in self.action_word.occurrences
        )
        if self.action_occurrences != expected_occurrences:
            raise ValueError("finite reachability rewrites ordered native action occurrences")
        if any(o.object_schema != ActionOccurrence.SCHEMA for o in self.action_occurrences):
            raise ValueError("finite reachability requires action occurrence identities")
        available = set(self.response_set.joint_calibration.coordinates)
        if any(i.coordinate not in available for i in self.task.boxes[0].intervals):
            raise ValueError("task consumes an unqualified readout, unit, frame or clock")


class FiniteCertificateStatus(StrEnum):
    CERTIFIED = "CERTIFIED"
    NOT_CERTIFIED = "NOT_CERTIFIED"
    UNEVALUABLE = "UNEVALUABLE"
    OUTSIDE_SUPPORT = "OUTSIDE_SUPPORT"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class FiniteConstraintMargin(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-constraint-margin'

    qualification_view_id: str
    box_id: str
    coordinate: FiniteResponseCoordinate
    lower_face_margin: Decimal
    upper_face_margin: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_view_id, field_name="qualification_view_id")
        validate_stable_id(self.box_id, field_name="box_id")
        validate_decimal(self.lower_face_margin, field_name="lower_face_margin")
        validate_decimal(self.upper_face_margin, field_name="upper_face_margin")


def finite_constraint_margins(
    request: FiniteResponseSetReachabilityRequest,
) -> tuple[FiniteConstraintMargin, ...]:
    """Project every face, including outward numerical uncertainty, in native units."""

    if request.response_set.disposition is not FiniteSetDisposition.AVAILABLE:
        return ()
    bounds = {(b.qualification_view_id, b.coordinate): b for b in request.response_set.bounds}
    return tuple(
        FiniteConstraintMargin(
            view,
            box.box_id,
            target.coordinate,
            bounds[view, target.coordinate].expanded_interval[0] - target.lower,
            target.upper - bounds[view, target.coordinate].expanded_interval[1],
        )
        for view in request.response_set.planned_coordinate.qualification_view_ids
        for box in request.task.boxes
        for target in box.intervals
    )


def finite_certificate_status(
    request: FiniteResponseSetReachabilityRequest,
) -> FiniteCertificateStatus:
    """Require containment in a union component for every qualification view."""

    disposition = request.response_set.disposition
    if disposition is not FiniteSetDisposition.AVAILABLE:
        return {
            FiniteSetDisposition.UNAVAILABLE: FiniteCertificateStatus.UNEVALUABLE,
            FiniteSetDisposition.OUTSIDE_SUPPORT: FiniteCertificateStatus.OUTSIDE_SUPPORT,
            FiniteSetDisposition.EXPIRED: FiniteCertificateStatus.EXPIRED,
        }[disposition]
    margins = finite_constraint_margins(request)
    for view in request.response_set.planned_coordinate.qualification_view_ids:
        if not any(
            all(
                m.lower_face_margin >= 0 and m.upper_face_margin >= 0
                for m in margins
                if m.qualification_view_id == view and m.box_id == box_id
            )
            for box_id in request.task.tie_order
        ):
            return FiniteCertificateStatus.NOT_CERTIFIED
    return FiniteCertificateStatus.CERTIFIED


@dataclass(frozen=True, slots=True)
class FiniteResponseSetReachabilityResult(CanonicalRecord):
    """Recomputable raw certificate; it neither admits an action nor issues authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-response-set-reachability-result'

    result_id: str
    request: FiniteResponseSetReachabilityRequest
    evaluator_implementation: ObjectIdentity
    margins: tuple[FiniteConstraintMargin, ...]
    status: FiniteCertificateStatus

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.margins != finite_constraint_margins(self.request):
            raise ValueError("finite certificate margins differ from the complete response set")
        if self.status is not finite_certificate_status(self.request):
            raise ValueError(
                "finite certificate status is not derived from complete-set containment"
            )


@dataclass(frozen=True, slots=True)
class FiniteReachabilityMethodReceipt(CanonicalRecord):
    """Evidence-bound finite certificate with no controlled-map rank semantics."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-reachability-method-receipt'

    receipt_id: str
    request: FiniteResponseSetReachabilityRequest
    raw_result: ObjectIdentity
    admission_direction_gate_receipt_ids: tuple[str, ...]
    horizon: HorizonSpec
    constraint_ids: tuple[str, ...]
    evidence_domain: ObjectIdentity
    authority_boundary: ObjectIdentity
    producer: ObjectIdentity
    resource_envelope: ObjectIdentity
    method: ExecutableReference
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    margins: tuple[FiniteConstraintMargin, ...]
    status: FiniteCertificateStatus

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_strings(
            self.admission_direction_gate_receipt_ids,
            field_name="admission_direction_gate_receipt_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.constraint_ids, field_name="constraint_ids", allow_empty=False
        )
        if self.raw_result.object_schema != (
            'empirical-lawhood/planning/finite-response-set-reachability-result'
        ):
            raise ValueError("finite admission receipt requires its exact finite-set result")
        expected_result = FiniteResponseSetReachabilityResult(
            result_id=f"finite-reachability.{self.request.request_id}",
            request=self.request,
            evaluator_implementation=ObjectIdentity.from_record(
                self.method.reference_id, self.method
            ),
            margins=self.margins,
            status=self.status,
        )
        if self.raw_result != ObjectIdentity.from_record(
            expected_result.result_id, expected_result
        ):
            raise ValueError(
                "finite admission receipt raw-result identity fails exact deterministic replay"
            )
        _validate_admission_receipt_evidence(
            evaluator=self.method,
            input_artifacts=self.input_artifacts,
            evidence_links=self.evidence_links,
        )
        source = self.request.response_set
        required_artifacts = (
            *source.source_artifacts,
            source.law_payload,
            source.joint_calibration.calibration_artifact,
            self.method.payload,
        )
        supplied = {a.artifact_id: a for a in self.input_artifacts}
        if any(supplied.get(a.artifact_id) != a for a in required_artifacts):
            raise ValueError("finite receipt loses exact payload/calibration/set artifacts")
        if any(
            link.information_cutoff_id != self.information_cutoff.cutoff_id
            or link.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or not link.visibility_ceiling.is_promotable
            for link in self.evidence_links
        ):
            raise ValueError("finite admission receipt evidence violates its causal/visibility boundary")
        if self.margins != finite_constraint_margins(self.request):
            raise ValueError("finite admission receipt rewrites native constraint margins")
        if self.status is not finite_certificate_status(self.request):
            raise ValueError("finite admission receipt status is not mechanically derived")

    @property
    def planned_coordinate(self) -> ReceiptAdmissionPlannedCoordinate:
        return self.request.response_set.planned_coordinate

    @property
    def information_cutoff(self) -> InformationCutoff:
        return self.request.information_cutoff

    @property
    def outcome_access(self) -> OutcomeAccess:
        return self.request.response_set.outcome_access

    @property
    def product_key(self) -> str:
        return self.planned_coordinate.coordinate_id
