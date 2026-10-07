"""Root-max operands for separate absolute and paired-feed uncertainty envelopes."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar
import numpy as np
from scipy.stats import beta
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.provenance import ObjectIdentity
from .config import FeedQualificationDesign, ROOTS
from .records import FeedRootEvidence


@dataclass(frozen=True, slots=True)
class FeedRootScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prepared-feed-qualification/feed-root-score'
    root: str
    domain: str
    receiver: int
    covered_rows: int
    action_assay: bool
    calibration_score: D | None
    coverage_score: D | None
    maximum_native_error: D | None
    invalidity: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.root not in {r for r, _, _, _ in ROOTS}
            or self.domain != "d11010"
            or self.receiver not in (0, 1)
            or self.covered_rows not in (0, 1)
            or self.action_assay != bool(self.covered_rows)
        ):
            raise ValueError("invalid prepared feed denominator")
        for value in (self.calibration_score, self.coverage_score, self.maximum_native_error):
            if value is not None and (not value.is_finite() or value < 0):
                raise ValueError("invalid prepared feed error operand")
        if not self.action_assay and any(
            v is not None
            for v in (self.calibration_score, self.coverage_score, self.maximum_native_error)
        ):
            raise ValueError("no contact cannot have a score")
        if tuple(sorted(set(self.invalidity))) != self.invalidity:
            raise ValueError("feed invalidity census differs")


def root_scores(
    design: FeedQualificationDesign, evidence: FeedRootEvidence
) -> tuple[FeedRootScore, ...]:
    if evidence.recipe != ObjectIdentity.from_record(design.config_id, design):
        raise ValueError("feed evidence changed its issued specification")
    domain = design.atlas.domains[0]
    _, policy, callback, actions = evidence.assays[0]
    contact = callback is not None
    invalid = {reason for _, reason in evidence.failures}
    scores: list[tuple[D, D, D] | None] = [None, None]
    if contact:
        assert callback is not None and policy == 0
        arrays = evidence.arrays.unpack()
        keys = [(a, v, f"d11010_a{a}_v{v}") for a in actions for v in (0, 1)]
        if any(f"{key}_grid" not in arrays for _, _, key in keys):
            invalid.add("MISSING_ASSIGNED_FEED_ASSAY")
        else:
            x = arrays["d11010_features"][list(actions)]
            if not domain.proposed_support(x, np.asarray(actions)).all():
                raise ValueError("native anchor is outside declared feed support")
            prediction = domain.predict(x)
            labels = np.zeros((3, 2))
            for a, v, key in keys:
                grid = arrays[f"{key}_grid"]
                times = np.arange(10 * 2**v + 1) / 2**v + callback * 10
                if (
                    grid.shape != (len(times), 8)
                    or not np.array_equal(grid[:, 0], times)
                    or not np.isfinite(grid).all()
                ):
                    invalid.add("ASSAY_GRID_INVALID")
                    continue
                labels[actions.index(a), v] = grid[:, 1].max()
                if not arrays[f"{key}_valid"].all():
                    invalid.add("ASSAY_DELIVERY_INVALID")
                exposure = arrays[f"{key}_exposure"]
                previous = arrays["exploration_unshifted_v0_stages"][callback - 1, 2:]
                if (
                    previous[0] > 0.016
                    or not np.all(exposure[:, 3] == previous[1])
                    or (a == 1 and np.any(exposure[:, 2] != 0))
                ):
                    invalid.add("FEED_PREPARATION_OR_FIXED_JACKET_INVALID")
            cooling = labels[0:1] - labels
            if np.any(np.abs(labels[:, 0] - labels[:, 1]) > 0.01):
                invalid.add("ABSOLUTE_NUMERICAL_INVALID")
            if np.any(np.abs(cooling[:, 0] - cooling[:, 1]) > 0.000001):
                invalid.add("CONTRAST_NUMERICAL_INVALID")
            absolute = np.abs(prediction[:, :1] - labels)
            response = np.abs(prediction[:, 1:2] - cooling)
            scale = 0.00005 + 0.5 * np.abs(prediction[:, 1:2])
            for receiver, error, divisor, padding in (
                (0, absolute, 0.25, 0.01),
                (1, response, scale, 0.000001),
            ):
                values = (
                    np.max(error / divisor),
                    np.max(np.maximum(error - padding, 0) / divisor),
                    np.max(error),
                )
                scores[receiver] = (
                    D(repr(float(values[0]))),
                    D(repr(float(values[1]))),
                    D(repr(float(values[2]))),
                )
    result = []
    for r in (0, 1):
        score = scores[r]
        result.append(
            FeedRootScore(
                evidence.root,
                domain.domain_id,
                r,
                int(contact),
                contact,
                None if score is None else score[0],
                None if score is None else score[1],
                None if score is None else score[2],
                tuple(sorted(invalid)),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FeedCalibrationCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prepared-feed-qualification/feed-calibration-cell'
    domain: str
    receiver: int
    roots: tuple[str, ...]
    q: D | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.domain != "d11010"
            or self.receiver not in (0, 1)
            or tuple(sorted(set(self.roots))) != self.roots
            or not set(self.roots) <= {r for r, role, _, _ in ROOTS if role == "calibration"}
            or (self.q is not None and (not self.q.is_finite() or self.q < 0))
            or (not self.reasons and (len(self.roots) < 29 or self.q is None or self.q > 1))
        ):
            raise ValueError("invalid feed calibration")


@dataclass(frozen=True, slots=True)
class FeedCalibration(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prepared-feed-qualification/feed-calibration'
    recipe: ObjectIdentity
    evidence: tuple[ObjectIdentity, ...]
    scores: tuple[FeedRootScore, ...]
    cells: tuple[FeedCalibrationCell, ...]

    def __post_init__(self) -> None:
        expected = tuple(r for r, role, _, _ in ROOTS if role == "calibration")
        if (
            tuple(e.object_id for e in self.evidence) != expected
            or len(self.scores) != 64
            or tuple((c.domain, c.receiver) for c in self.cells) != (("d11010", 0), ("d11010", 1))
        ):
            raise ValueError("feed calibration census differs")
        for c in self.cells:
            selected = tuple(s for s in self.scores if s.receiver == c.receiver)
            if tuple(s.root for s in selected) != expected or c.roots != tuple(
                s.root for s in selected if s.action_assay
            ):
                raise ValueError("feed calibration contact denominator differs")


def calibration(
    design: FeedQualificationDesign, evidence: tuple[FeedRootEvidence, ...]
) -> FeedCalibration:
    if tuple(r.root for r in evidence) != tuple(
        r for r, role, _, _ in ROOTS if role == "calibration"
    ):
        raise ValueError("complete ordered calibration panel required")
    scores = tuple(s for r in evidence for s in root_scores(design, r))
    cells = []
    for receiver in (0, 1):
        selected = tuple(s for s in scores if s.receiver == receiver)
        contact = tuple(s for s in selected if s.action_assay)
        reasons = {r for s in selected for r in s.invalidity}
        if len(contact) < 29:
            reasons.add("INSUFFICIENT_CAUSAL_DOMAIN_CONTACT")
        if any(s.calibration_score is None for s in contact):
            reasons.add("MISSING_FEED_CALIBRATION_OPERAND")
        q = (
            None
            if reasons
            else max(s.calibration_score for s in contact if s.calibration_score is not None)
        )
        if q is not None and q > 1:
            reasons.add(
                "FEED_RESPONSE_PRECISION_FAILED" if receiver else "FEED_ABSOLUTE_PRECISION_FAILED"
            )
        cells.append(
            FeedCalibrationCell(
                "d11010", receiver, tuple(s.root for s in contact), q, tuple(sorted(reasons))
            )
        )
    return FeedCalibration(
        ObjectIdentity.from_record(design.config_id, design),
        tuple(ObjectIdentity.from_record(r.root, r) for r in evidence),
        scores,
        tuple(cells),
    )


@dataclass(frozen=True, slots=True)
class FeedQualificationCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prepared-feed-qualification/feed-qualification-cell'
    domain: str
    receiver: int
    contacted_roots: tuple[str, ...]
    adequate_roots: tuple[str, ...]
    lower_bound: D
    halfwidth: D | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = {r for r, role, _, _ in ROOTS if role == "qualification"}
        if (
            self.domain != "d11010"
            or self.receiver not in (0, 1)
            or not set(self.adequate_roots) <= set(self.contacted_roots) <= expected
            or tuple(sorted(set(self.contacted_roots))) != self.contacted_roots
            or tuple(sorted(set(self.adequate_roots))) != self.adequate_roots
            or not self.lower_bound.is_finite()
            or not 0 <= self.lower_bound <= 1
            or (
                self.halfwidth is not None
                and (not self.halfwidth.is_finite() or self.halfwidth < 0)
            )
            or (
                not self.reasons
                and (
                    len(self.contacted_roots) < 29
                    or self.lower_bound < D(".90")
                    or self.halfwidth is None
                )
            )
        ):
            raise ValueError("feed qualification operands differ")


@dataclass(frozen=True, slots=True)
class FeedQualificationOperands(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prepared-feed-qualification/feed-qualification-operands'
    recipe: ObjectIdentity
    calibration: ObjectIdentity
    evidence: tuple[ObjectIdentity, ...]
    heldout_scores: tuple[FeedRootScore, ...]
    cells: tuple[FeedQualificationCell, ...]

    def __post_init__(self) -> None:
        if (
            tuple(e.object_id for e in self.evidence) != tuple(r for r, _, _, _ in ROOTS)
            or len(self.heldout_scores) != 64
            or tuple((c.domain, c.receiver) for c in self.cells) != (("d11010", 0), ("d11010", 1))
        ):
            raise ValueError("feed qualification census differs")
        for c in self.cells:
            if tuple(s.root for s in self.heldout_scores if s.receiver == c.receiver) != tuple(
                r for r, role, _, _ in ROOTS if role == "qualification"
            ):
                raise ValueError("feed qualification lost an assigned root")


def qualification_cells(
    design: FeedQualificationDesign,
    calibrated: FeedCalibration,
    scores: tuple[FeedRootScore, ...],
) -> tuple[FeedQualificationCell, ...]:
    if calibrated.recipe != ObjectIdentity.from_record(design.config_id, design):
        raise ValueError("frozen feed calibration differs")
    cells = []
    for c in calibrated.cells:
        selected = tuple(s for s in scores if s.receiver == c.receiver)
        if tuple(s.root for s in selected) != tuple(
            r for r, role, _, _ in ROOTS if role == "qualification"
        ):
            raise ValueError("complete qualification panel required")
        contact = tuple(s for s in selected if s.action_assay)
        adequate = tuple(
            s.root
            for s in contact
            if not s.invalidity
            and c.q is not None
            and s.coverage_score is not None
            and s.coverage_score <= c.q
        )
        lower = (
            D(0)
            if not adequate
            else D(repr(float(beta.ppf(0.05, len(adequate), len(contact) - len(adequate) + 1))))
        )
        reasons = set(c.reasons) | {r for s in selected for r in s.invalidity}
        if len(contact) < 29:
            reasons.add("INSUFFICIENT_CAUSAL_DOMAIN_CONTACT")
        if lower < D(".90"):
            reasons.add("FEED_HELDOUT_ADEQUACY_FAILED")
        width = (
            None
            if c.q is None
            else c.q * (D(".25") if c.receiver == 0 else D(".00505"))
            + design.numerical_padding[c.receiver]
        )
        cells.append(
            FeedQualificationCell(
                "d11010",
                c.receiver,
                tuple(s.root for s in contact),
                adequate,
                lower,
                width,
                tuple(sorted(reasons)),
            )
        )
    return tuple(cells)
