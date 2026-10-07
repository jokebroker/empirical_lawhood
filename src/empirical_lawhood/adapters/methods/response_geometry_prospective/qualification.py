"""Finite assay response-contact records and outcome-blind reduction rules.

These are excluded assay diagnostics, never a ResponseLaw, controller or
protected result. The registered runtime adjudication owner retains finality.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from scipy.stats import beta

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id, validate_sha256
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import ResponseGeometryAssayNativeConfig, ResponseGeometryAssayNativeRoot, ResponseGeometryAssayNativeSegmentResult, PARENTS, assay_roots, assay_segments


def parent_work_failure(work: Decimal | None) -> str | None:
    """frozen common assay's absolute generalized density-work bound, separate from inner effort."""
    if work is None:
        return "PARENT_DENSITY_WORK_UNRESOLVED"
    return "PARENT_DENSITY_WORK_LIMIT_EXCEEDED" if work > Decimal(32) else None


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayProjectionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-assay-projection-config'
    config_id: str
    source_config: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.source_config.object_schema != ResponseGeometryAssayNativeConfig.SCHEMA:
            raise ValueError("assay projection requires its exact native config identity")


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayEvaluationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-assay-evaluation-config'
    config_id: str
    projection_config: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.projection_config.object_schema != ResponseGeometryAssayProjectionConfig.SCHEMA:
            raise ValueError("assay evaluator requires its exact projection config identity")


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayBoundary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-assay-boundary'
    sign: int
    tick: int
    before: str
    after: str

    def __post_init__(self) -> None:
        if type(self.sign) is not int or self.sign not in {-1, 0, 1}:
            raise ValueError("assay boundary changes its branch sign")
        if type(self.tick) is not int or self.tick < 16 or self.tick % 16:
            raise ValueError("assay boundary must retain its 16-tick interval censoring")
        if (
            self.before not in {"G", "N"}
            or self.after not in {"G", "N"}
            or self.before == self.after
        ):
            raise ValueError("assay boundary requires two measured different states")


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayPacket(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-assay-packet'
    parent: str
    delivered: bool
    contact: bool | None
    preservation: bool | None
    signed_responses: tuple[Decimal, ...]
    trace_responses: tuple[Decimal, ...]
    odd_response: Decimal | None
    even_response: Decimal | None
    free_damped_response: Decimal
    maximum_orthogonal_x: Decimal | None
    maximum_relative_y: Decimal | None
    maximum_transfer_difference: Decimal | None
    geometry_labels: tuple[str, ...]
    boundaries: tuple[ResponseGeometryAssayBoundary, ...]
    force_work: tuple[Decimal, ...]
    parent_absolute_density_work: Decimal | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.parent not in PARENTS or type(self.delivered) is not bool:
            raise ValueError("assay packet changes its declared parent/delivery role")
        for predicate in (self.contact, self.preservation):
            if predicate is not None and type(predicate) is not bool:
                raise ValueError("assay predicates must distinguish measured Boolean from unknown")
        for values in (self.signed_responses, self.trace_responses, self.force_work):
            if len(values) not in {0, 3} or any(not value.is_finite() for value in values):
                raise ValueError("assay packet must retain the complete finite NEG/HOLD/POS vector")
        for value in (
            self.odd_response,
            self.even_response,
            self.free_damped_response,
            self.maximum_orthogonal_x,
            self.maximum_relative_y,
            self.maximum_transfer_difference,
            self.parent_absolute_density_work,
        ):
            if value is not None and not value.is_finite():
                raise ValueError("assay packet numerical operands must be finite or unknown")
        if len(self.geometry_labels) != 5 or any(
            value not in {"G", "N", "U"} for value in self.geometry_labels
        ):
            raise ValueError("assay packet requires pre-parent, invocation and three terminal labels")
        if len(self.boundaries) > 1024 or tuple(sorted(set(self.reasons))) != self.reasons:
            raise ValueError("assay packet boundary/reason roster differs")
        if self.contact is not None and (not self.delivered or len(self.signed_responses) != 3):
            raise ValueError("assay measured contact requires its complete native response vector")
        if self.parent_absolute_density_work is not None and self.parent_absolute_density_work < 0:
            raise ValueError("assay absolute parent density work cannot be negative")

    @property
    def parent_work_admissible(self) -> bool | None:
        if self.parent_absolute_density_work is None:
            return None
        return parent_work_failure(self.parent_absolute_density_work) is None

    @property
    def qualification_reasons(self) -> tuple[str, ...]:
        failure = parent_work_failure(self.parent_absolute_density_work)
        return tuple(sorted(set(self.reasons) | ({failure} if failure is not None else set())))

    @property
    def positive(self) -> bool:
        return (
            self.delivered
            and self.contact is True
            and self.preservation is True
            and self.parent_work_admissible is True
        )

    @property
    def decision_vector(self) -> tuple[object, ...]:
        return (
            self.delivered,
            self.contact,
            self.preservation,
            self.parent_work_admissible,
            self.geometry_labels,
        )


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayViewReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-assay-view-report'
    report_id: str
    projection_config: ObjectIdentity
    root: ResponseGeometryAssayNativeRoot
    refinement: int
    source_results: tuple[ObjectIdentity, ...]
    primary_mode_resolved: bool
    shadow_mode_distance: Decimal | None
    packets: tuple[ResponseGeometryAssayPacket, ...]
    diagnostics_sha256: str

    def __post_init__(self) -> None:
        if self.report_id != f"report.{self.root.root_id}.r{self.refinement}":
            raise ValueError("assay view report identity differs from its root/view")
        if self.projection_config.object_schema != ResponseGeometryAssayProjectionConfig.SCHEMA:
            raise ValueError("assay report lacks its exact projection config")
        if type(self.refinement) is not int or self.refinement not in self.root.refinements:
            raise ValueError("assay report adds an undeclared numerical view")
        if type(self.primary_mode_resolved) is not bool:
            raise ValueError("assay primary-mode status must be explicit")
        if self.shadow_mode_distance is not None and (
            not self.shadow_mode_distance.is_finite() or self.shadow_mode_distance < 0
        ):
            raise ValueError("assay shadow-mode distance must be a finite nonnegative diagnostic")
        if tuple(value.parent for value in self.packets) != PARENTS:
            raise ValueError("assay report drops or adds a parent packet")
        expected = tuple(
            f"result.{segment.task_id}" for segment in assay_segments() if segment.root == self.root
        )
        if tuple(value.object_id for value in self.source_results) != expected or any(
            value.object_schema != ResponseGeometryAssayNativeSegmentResult.SCHEMA for value in self.source_results
        ):
            raise ValueError("assay source lineage must contain every exact ordered root segment")
        validate_sha256(self.diagnostics_sha256, field_name="diagnostics_sha256")

    @property
    def mode_qualified(self) -> bool:
        return (
            self.primary_mode_resolved
            and self.shadow_mode_distance is not None
            and self.shadow_mode_distance <= Decimal("0.05")
        )


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayProportion(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-assay-proportion'
    count: int
    total: int
    alpha: Decimal
    lower: Decimal
    upper: Decimal

    def __post_init__(self) -> None:
        if (
            type(self.count) is not int
            or type(self.total) is not int
            or not 0 <= self.count <= self.total
            or self.total < 1
        ):
            raise ValueError("assay proportion requires a positive fixed denominator and valid count")
        if (
            not all(value.is_finite() for value in (self.alpha, self.lower, self.upper))
            or not 0 < self.alpha < 1
            or not 0 <= self.lower <= self.upper <= 1
            or not self.lower <= Decimal(self.count) / self.total <= self.upper
        ):
            raise ValueError("assay proportion interval or error allocation differs")


def _proportion(count: int, total: int, alpha: Decimal) -> ResponseGeometryAssayProportion:
    return ResponseGeometryAssayProportion(
        count,
        total,
        alpha,
        Decimal(0)
        if count == 0
        else Decimal(str(float(beta.ppf(float(alpha) / 2, count, total - count + 1)))),
        Decimal(1)
        if count == total
        else Decimal(str(float(beta.ppf(1 - float(alpha) / 2, count + 1, total - count)))),
    )


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayParentContact(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-assay-parent-contact'
    parent: str
    dual_positive: ResponseGeometryAssayProportion
    positive_union: int
    jaccard: Decimal | None
    eligible: bool

    def __post_init__(self) -> None:
        if (
            self.parent not in PARENTS
            or self.dual_positive.total != 4
            or self.dual_positive.alpha != Decimal("0.05") / 40
            or type(self.positive_union) is not int
            or not self.dual_positive.count <= self.positive_union <= 4
        ):
            raise ValueError("assay parent contact changes the fixed four-root panel")
        ratio = (
            Decimal(self.dual_positive.count) / self.positive_union if self.positive_union else None
        )
        if (
            self.jaccard != ratio
            or type(self.eligible) is not bool
            or self.eligible
            != (self.dual_positive.count >= 3 and ratio is not None and ratio >= Decimal("0.9"))
        ):
            raise ValueError("assay parent eligibility differs from its positive intersection/union")


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-assay-cell'
    context: str
    landmark_tick: int
    assay: str
    agreement: ResponseGeometryAssayProportion
    mode_qualified: bool
    timing_tested: bool
    timing_agrees: bool | None
    maximum_boundary_difference: int | None
    parents: tuple[ResponseGeometryAssayParentContact, ...]
    fixed_origin_qualified: bool

    def __post_init__(self) -> None:
        if (
            self.context not in {"assembling", "prepared"}
            or type(self.landmark_tick) is not int
            or self.landmark_tick not in {1024, 4096}
            or self.assay not in {"short-pulse-response", "extended-pulse-response"}
            or self.agreement.total != 4
            or self.agreement.alpha != Decimal("0.05") / 8
            or tuple(value.parent for value in self.parents) != PARENTS
        ):
            raise ValueError("assay cell changes its fixed context/landmark/assay panel")
        if any(
            type(value) is not bool
            for value in (self.mode_qualified, self.timing_tested, self.fixed_origin_qualified)
        ) or (self.timing_tested != (self.timing_agrees is not None)):
            raise ValueError("assay cell must keep tested and untested timing distinct")
        if self.timing_agrees is not None and type(self.timing_agrees) is not bool:
            raise ValueError("assay timing agreement must be Boolean or untested")
        difference = self.maximum_boundary_difference
        if difference is not None and (
            type(difference) is not int
            or difference < 0
            or not self.timing_tested
            or (difference > 16 and self.timing_agrees is True)
        ):
            raise ValueError("assay timing comparison differs from its interval-censored boundary")
        if self.fixed_origin_qualified != (
            self.numerically_qualified and any(value.eligible for value in self.parents)
        ):
            raise ValueError("assay cell qualification differs from its noncompensating predicates")

    @property
    def numerically_qualified(self) -> bool:
        return (
            self.agreement.count == 4
            and self.mode_qualified
            and (not self.timing_tested or self.timing_agrees is True)
        )


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-assay-evaluation'
    evaluation_id: str
    evaluation_config: ObjectIdentity
    reports: tuple[ObjectIdentity, ...]
    cells: tuple[ResponseGeometryAssayCell, ...]
    selected_assay: str | None
    selected_landmarks: tuple[int, ...]
    terminal: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        if self.evaluation_config.object_schema != ResponseGeometryAssayEvaluationConfig.SCHEMA:
            raise ValueError("assay terminal requires its exact evaluation config")
        expected = tuple(
            (context, tick, assay)
            for context in ("assembling", "prepared")
            for tick in (1024, 4096)
            for assay in ("short-pulse-response", "extended-pulse-response")
        )
        if tuple((cell.context, cell.landmark_tick, cell.assay) for cell in self.cells) != expected:
            raise ValueError("assay terminal drops or adds a context/landmark/assay cell")
        if self.terminal not in {
            "FIXED_ORIGIN_QUALIFIED",
            "NUMERICAL_OR_SEMANTIC_SUPPORT_UNRESOLVED",
            "INNER_ACTION_CHANNEL_UNQUALIFIED",
        }:
            raise ValueError("assay terminal exceeds its finite assay role")
        if (self.selected_assay is not None) != (self.terminal == "FIXED_ORIGIN_QUALIFIED"):
            raise ValueError("assay selected assay and terminal disagree")
        if self.selected_assay not in {None, "short-pulse-response", "extended-pulse-response"} or len(self.selected_landmarks) != (
            0 if self.selected_assay is None else 2
        ):
            raise ValueError("assay selected context/assay mapping differs")
        selection = _select(self.cells, contact=True)
        if (self.selected_assay, self.selected_landmarks) != selection:
            raise ValueError("assay selection changes the earliest-landmark/common-assay rule")
        if self.terminal != _terminal(self.cells, self.selected_assay):
            raise ValueError("assay terminal collapses numerical support and native action contact")
        expected_reports = tuple(
            sorted(
                f"report.{root.root_id}.r{refinement}"
                for root in assay_roots()
                for refinement in root.refinements
            )
        )
        if tuple(value.object_id for value in self.reports) != expected_reports or any(
            value.object_schema != ResponseGeometryAssayViewReport.SCHEMA for value in self.reports
        ):
            raise ValueError("assay terminal must retain all 72 exact report identities")
        if tuple(sorted(set(self.reasons))) != self.reasons:
            raise ValueError("assay terminal reasons must be sorted and unique")


def _select(cells: tuple[ResponseGeometryAssayCell, ...], *, contact: bool) -> tuple[str | None, tuple[int, ...]]:
    for assay in ("short-pulse-response", "extended-pulse-response"):
        available = tuple(
            tuple(
                cell.landmark_tick
                for cell in cells
                if cell.context == context
                and cell.assay == assay
                and (cell.fixed_origin_qualified if contact else cell.numerically_qualified)
            )
            for context in ("assembling", "prepared")
        )
        if all(available):
            return assay, tuple(min(values) for values in available)
    return None, ()


def _terminal(cells: tuple[ResponseGeometryAssayCell, ...], selected: str | None) -> str:
    if selected is not None:
        return "FIXED_ORIGIN_QUALIFIED"
    if _select(cells, contact=False)[0] is None:
        return "NUMERICAL_OR_SEMANTIC_SUPPORT_UNRESOLVED"
    return "INNER_ACTION_CHANNEL_UNQUALIFIED"


def evaluate_response_geometry_assay(config: ResponseGeometryAssayEvaluationConfig, reports: tuple[ResponseGeometryAssayViewReport, ...]) -> ResponseGeometryAssayEvaluation:
    expected = {(root, refinement) for root in assay_roots() for refinement in root.refinements}
    by_view = {(value.root, value.refinement): value for value in reports}
    if (
        len(by_view) != len(reports)
        or set(by_view) != expected
        or any(value.projection_config != config.projection_config for value in reports)
    ):
        raise ValueError("assay evaluator requires all 72 exact root/view reports and their config")
    cells = []
    for context in ("assembling", "prepared"):
        for tick in (1024, 4096):
            for assay in ("short-pulse-response", "extended-pulse-response"):
                roots = tuple(
                    root
                    for root in assay_roots()
                    if (root.context, root.landmark_tick, root.assay) == (context, tick, assay)
                )
                pairs = tuple((by_view[(root, 1)], by_view[(root, 2)]) for root in roots)
                agreement = sum(
                    all(
                        a.delivered
                        and b.delivered
                        and a.contact is not None
                        and b.contact is not None
                        and a.preservation is not None
                        and b.preservation is not None
                        and a.parent_work_admissible is not None
                        and b.parent_work_admissible is not None
                        and a.decision_vector == b.decision_vector
                        for a, b in zip(left.packets, right.packets, strict=True)
                    )
                    for left, right in pairs
                )
                modes = all(left.mode_qualified and right.mode_qualified for left, right in pairs)
                differences: list[int] = []
                timing_agrees, timing_tested = True, False
                parents = []
                for index, parent in enumerate(PARENTS):
                    positive_left = {
                        root.root_id
                        for root, (left, _) in zip(roots, pairs, strict=True)
                        if left.packets[index].positive and left.mode_qualified
                    }
                    positive_right = {
                        root.root_id
                        for root, (_, right) in zip(roots, pairs, strict=True)
                        if right.packets[index].positive and right.mode_qualified
                    }
                    common, union = (
                        len(positive_left & positive_right),
                        len(positive_left | positive_right),
                    )
                    ratio = None if not union else Decimal(common) / union
                    parents.append(
                        ResponseGeometryAssayParentContact(
                            parent,
                            _proportion(common, 4, Decimal("0.05") / 40),
                            union,
                            ratio,
                            common >= 3 and ratio is not None and ratio >= Decimal("0.9"),
                        )
                    )
                    for left, right in pairs:
                        first, second = (
                            left.packets[index].boundaries,
                            right.packets[index].boundaries,
                        )
                        timing_tested |= bool(first or second)
                        if tuple((b.sign, b.before, b.after) for b in first) != tuple(
                            (b.sign, b.before, b.after) for b in second
                        ):
                            timing_agrees = False
                        else:
                            differences.extend(
                                abs(a.tick - b.tick) for a, b in zip(first, second, strict=True)
                            )
                if differences and max(differences) > 16:
                    timing_agrees = False
                cells.append(
                    ResponseGeometryAssayCell(
                        context,
                        tick,
                        assay,
                        _proportion(agreement, 4, Decimal("0.05") / 8),
                        modes,
                        timing_tested,
                        timing_agrees if timing_tested else None,
                        max(differences) if differences else None,
                        tuple(parents),
                        agreement == 4
                        and modes
                        and any(value.eligible for value in parents)
                        and (not timing_tested or timing_agrees),
                    )
                )
    selected, landmarks = _select(tuple(cells), contact=True)
    reasons = {
        reason
        for report in reports
        for packet in report.packets
        for reason in packet.qualification_reasons
    }
    if any(not cell.timing_tested for cell in cells):
        reasons.add("TIMING_UNTESTED_IN_SOME_CELLS")
    if any(not cell.mode_qualified for cell in cells):
        reasons.add("MODE_UNQUALIFIED_IN_SOME_CELLS")
    if selected is None:
        reasons.add("NO_COMMON_QUALIFIED_ASSAY")
    else:
        reasons.add("FIXED_ORIGIN_ASSAY_QUALIFIED")
    return ResponseGeometryAssayEvaluation(
        "response-geometry-assay.evaluation",
        ObjectIdentity.from_record(config.config_id, config),
        tuple(
            sorted(
                (ObjectIdentity.from_record(value.report_id, value) for value in reports),
                key=lambda value: value.object_id,
            )
        ),
        tuple(cells),
        selected,
        landmarks,
        _terminal(tuple(cells), selected),
        tuple(sorted(reasons)),
    )
