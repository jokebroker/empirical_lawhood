"""Receiver/domain operands from assigned fresh roots; no law verdicts here."""

from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import EXPLORATION_ARRAY_STEMS
from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

import numpy as np
from scipy.stats import beta

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation
from empirical_lawhood.adapters.simulators.reactor_causal_response.projection import project_episode
from .config import LocalQualificationDesign, ROOTS
from .records import LocalRootEvidence


@dataclass(frozen=True, slots=True)
class LocalRootScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-local-domain-qualification/local-root-score'
    root: str
    domain: str
    receiver: int
    covered_rows: int
    action_assay: bool
    absolute_error: D | None
    response_error: D | None
    zero_response_error: D | None
    invalidity: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.domain, field_name="domain")
        if (
            self.root not in {r for r, _, _, _ in ROOTS}
            or self.receiver not in (0, 1)
            or not 0 <= self.covered_rows <= 8640
        ):
            raise ValueError("invalid local root score denominator")
        for value in (self.absolute_error, self.response_error, self.zero_response_error):
            if value is not None and (not value.is_finite() or value < 0):
                raise ValueError("invalid local error measurement")
        if self.covered_rows == 0 and self.absolute_error is not None:
            raise ValueError("no local observation cannot receive a predictive score")
        if not self.action_assay and (
            self.response_error is not None or self.zero_response_error is not None
        ):
            raise ValueError("no action assay cannot receive a response score")
        if tuple(sorted(set(self.invalidity))) != self.invalidity:
            raise ValueError("local invalidity census differs")


def root_scores(
    design: LocalQualificationDesign, evidence: LocalRootEvidence
) -> tuple[LocalRootScore, ...]:
    if evidence.recipe != ObjectIdentity.from_record(design.config_id, design):
        raise ValueError("root changed the frozen local qualification design")
    if tuple(x[0] for x in evidence.assays) != tuple(d.domain_id for d in design.atlas.candidates):
        raise ValueError("missing assigned domain assay")
    arrays = evidence.arrays.unpack()
    donors = []
    donor_missing = any(key.startswith("e") for key, _ in evidence.failures)
    for donor_policy in range(3):
        keys = tuple(f"{EXPLORATION_ARRAY_STEMS[donor_policy]}_v{view}" for view in (0, 1))
        if any(f"{key}_labels" not in arrays for key in keys):
            donor_missing = True
            continue
        observations = tuple(
            Observation(*map(float, row)) for row in arrays[f"{keys[0]}_observations"]
        )
        requests = tuple((float(row[0]), float(row[1])) for row in arrays[f"{keys[0]}_requests"])
        x, donor_action_ids = project_episode(observations, requests)
        labels = np.stack([arrays[f"{key}_labels"] for key in keys], axis=1)
        valid = np.logical_and.reduce([arrays[f"{key}_valid"].astype(bool) for key in keys])
        valid &= (np.abs(labels[:, 0] - labels[:, 1]) <= (0.01, 0.0002, 0.000001)).all(axis=1)
        if not np.array_equal(arrays[f"{keys[0]}_requests"], arrays[f"{keys[1]}_requests"]):
            valid[:] = False
        donors.append((donor_policy, x, donor_action_ids, labels, valid))
    result = []
    for domain in design.atlas.candidates:
        errors = []
        invalid = {"MISSING_ASSIGNED_DONOR"} if donor_missing else set()
        covered = 0
        for _, x, donor_action_ids, labels, valid in donors:
            mask = domain.proposed_support(x, donor_action_ids)
            covered += int(mask.sum())
            if mask.any():
                errors.append(
                    np.abs(domain.predict(x[mask])[:, None] - labels[mask, :, :2]).reshape(-1, 2)
                )
                if not valid[mask].all():
                    invalid.add("LOCAL_NUMERICAL_OR_DELIVERY_INVALID")
        _, policy, callback, actions = next(a for a in evidence.assays if a[0] == domain.domain_id)
        responses, zeros = [], []
        has_assay = policy is not None and callback is not None
        if has_assay:
            assert callback is not None
            required = tuple((action, view) for action in actions for view in (0, 1))
            if any(f"{domain.domain_id}_a{a}_v{v}_grid" not in arrays for a, v in required):
                invalid.add("MISSING_ASSIGNED_ACTION_ASSAY")
            else:
                x = arrays[f"{domain.domain_id}_features"]
                predicted = domain.predict(x)
                donor = next((p for p in donors if p[0] == policy), None)
                if donor is None:
                    invalid.add("MISSING_ASSAY_DONOR")
                else:
                    _, donor_x, donor_actions, donor_labels, _ = donor
                    donor_prediction = domain.predict(donor_x[callback : callback + 1])[0]
                    for action in actions:
                        pair = []
                        for view in (0, 1):
                            key = f"{domain.domain_id}_a{action}_v{view}"
                            grid = arrays[f"{key}_grid"]
                            expected_times = np.arange(10 * 2**view + 1) / 2**view + callback * 10
                            if (
                                grid.shape != (len(expected_times), 8)
                                or not np.array_equal(grid[:, 0], expected_times)
                                or not np.isfinite(grid).all()
                            ):
                                invalid.add("ASSAY_GRID_INVALID")
                                continue
                            label = np.array((grid[:, 1].max(), grid[-1, 4], grid[-1, 3]))
                            pair.append(label)
                            errors.append(np.abs(predicted[action] - label[:2])[None])
                            actual_response = label[:2] - donor_labels[callback, view, :2]
                            response_error = np.abs(
                                predicted[action] - donor_prediction - actual_response
                            )
                            responses.append(response_error)
                            zeros.append(np.abs(actual_response))
                            if not arrays[f"{key}_valid"].all() or any(
                                k == key for k, _ in evidence.failures
                            ):
                                invalid.add("ASSAY_DELIVERY_OR_PREFIX_INVALID")
                        if len(pair) != 2 or np.any(
                            np.abs(pair[0] - pair[1]) > (0.01, 0.0002, 0.000001)
                        ):
                            invalid.add("ASSAY_NUMERICAL_INVALID")
        if any(not np.isfinite(a).all() for a in (*errors, *responses, *zeros)):
            invalid.add("NONFINITE_LOCAL_MEASUREMENT_INVALID")
            errors, responses, zeros = [], [], []
        maximum = np.concatenate(errors).max(axis=0) if errors else None
        response = np.asarray(responses).max(axis=0) if responses else None
        zero = np.asarray(zeros).max(axis=0) if zeros else None
        for receiver in domain.nominated_receivers:
            result.append(
                LocalRootScore(
                    evidence.root,
                    domain.domain_id,
                    receiver,
                    covered,
                    has_assay,
                    None if maximum is None else D(repr(float(maximum[receiver]))),
                    None if response is None else D(repr(float(response[receiver]))),
                    None if zero is None else D(repr(float(zero[receiver]))),
                    tuple(sorted(invalid)),
                )
            )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class LocalCalibrationCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-local-domain-qualification/local-calibration-cell'
    domain: str
    receiver: int
    roots: tuple[str, ...]
    q: D | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.domain, field_name="domain")
        expected = {r for r, role, _, _ in ROOTS if role == "calibration"}
        if (
            self.receiver not in (0, 1)
            or tuple(sorted(set(self.roots))) != self.roots
            or not set(self.roots) <= expected
            or tuple(sorted(set(self.reasons))) != self.reasons
            or (self.q is not None and (not self.q.is_finite() or self.q < 0))
            or (not self.reasons and (len(self.roots) < 29 or self.q is None or self.q > 1))
        ):
            raise ValueError("invalid local calibration cell")


@dataclass(frozen=True, slots=True)
class LocalCalibration(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-local-domain-qualification/local-calibration'
    recipe: ObjectIdentity
    evidence: tuple[ObjectIdentity, ...]
    scores: tuple[LocalRootScore, ...]
    cells: tuple[LocalCalibrationCell, ...]

    def __post_init__(self) -> None:
        expected = tuple(r for r, role, _, _ in ROOTS if role == "calibration")
        if (
            tuple(e.object_id for e in self.evidence) != expected
            or len(self.cells) != 16
            or len(self.scores) != 512
            or len({(c.domain, c.receiver) for c in self.cells}) != 16
        ):
            raise ValueError("local calibration assigned census differs")
        for cell in self.cells:
            selected = tuple(
                s for s in self.scores if (s.domain, s.receiver) == (cell.domain, cell.receiver)
            )
            if tuple(s.root for s in selected) != expected or cell.roots != tuple(
                s.root for s in selected if s.covered_rows and s.action_assay
            ):
                raise ValueError("local calibration contact denominator differs")


def calibration(
    design: LocalQualificationDesign, evidence: tuple[LocalRootEvidence, ...]
) -> LocalCalibration:
    expected = tuple(r for r, role, _, _ in ROOTS if role == "calibration")
    if tuple(r.root for r in evidence) != expected:
        raise ValueError("complete ordered assigned calibration panel required")
    scores = tuple(score for root in evidence for score in root_scores(design, root))
    cells = []
    for domain in design.atlas.candidates:
        for receiver in domain.nominated_receivers:
            selected = tuple(
                s for s in scores if s.domain == domain.domain_id and s.receiver == receiver
            )
            contact = tuple(s for s in selected if s.covered_rows and s.action_assay)
            reasons = set(r for s in selected for r in s.invalidity)
            if len(contact) < design.minimum_contact_roots:
                reasons.add("INSUFFICIENT_CAUSAL_DOMAIN_CONTACT")
            if any(s.absolute_error is None or s.response_error is None for s in contact):
                reasons.add("MISSING_LOCAL_CALIBRATION_OPERAND")
            q = (
                None
                if reasons
                else max(
                    s.absolute_error / design.absolute_scale[receiver]
                    for s in contact
                    if s.absolute_error is not None
                )
            )
            if q is not None and q > 1:
                reasons.add("LOCAL_CALIBRATION_PRECISION_FAILED")
            if any(
                s.response_error is not None
                and s.response_error
                > design.response_scale[receiver] + 2 * design.numerical_padding[receiver]
                for s in contact
            ):
                reasons.add("LOCAL_ACTION_RESPONSE_PRECISION_FAILED")
            cells.append(
                LocalCalibrationCell(
                    domain.domain_id,
                    receiver,
                    tuple(s.root for s in contact),
                    q,
                    tuple(sorted(reasons)),
                )
            )
    return LocalCalibration(
        ObjectIdentity.from_record(design.config_id, design),
        tuple(ObjectIdentity.from_record(r.root, r) for r in evidence),
        scores,
        tuple(cells),
    )


@dataclass(frozen=True, slots=True)
class LocalQualificationCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-local-domain-qualification/local-qualification-cell'
    domain: str
    receiver: int
    contacted_roots: tuple[str, ...]
    adequate_roots: tuple[str, ...]
    lower_bound: D
    halfwidth: D | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.domain, field_name="domain")
        expected = {r for r, role, _, _ in ROOTS if role == "qualification"}
        if (
            self.receiver not in (0, 1)
            or tuple(sorted(set(self.contacted_roots))) != self.contacted_roots
            or tuple(sorted(set(self.adequate_roots))) != self.adequate_roots
            or not set(self.adequate_roots) <= set(self.contacted_roots) <= expected
            or not self.lower_bound.is_finite()
            or not 0 <= self.lower_bound <= 1
            or (
                self.halfwidth is not None
                and (not self.halfwidth.is_finite() or self.halfwidth < 0)
            )
            or tuple(sorted(set(self.reasons))) != self.reasons
            or (
                not self.reasons
                and (
                    len(self.contacted_roots) < 29
                    or self.lower_bound < D(".9")
                    or self.halfwidth is None
                )
            )
        ):
            raise ValueError("invalid local qualification cell")


@dataclass(frozen=True, slots=True)
class LocalQualificationOperands(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-local-domain-qualification/local-qualification-operands'
    recipe: ObjectIdentity
    calibration: ObjectIdentity
    evidence: tuple[ObjectIdentity, ...]
    heldout_scores: tuple[LocalRootScore, ...]
    cells: tuple[LocalQualificationCell, ...]

    def __post_init__(self) -> None:
        if (
            tuple(e.object_id for e in self.evidence) != tuple(r for r, _, _, _ in ROOTS)
            or len(self.heldout_scores) != 512
            or len(self.cells) != 16
            or len({(c.domain, c.receiver) for c in self.cells}) != 16
        ):
            raise ValueError("local qualification operand census differs")
        expected = tuple(r for r, role, _, _ in ROOTS if role == "qualification")
        for cell in self.cells:
            if (
                tuple(
                    s.root
                    for s in self.heldout_scores
                    if (s.domain, s.receiver) == (cell.domain, cell.receiver)
                )
                != expected
            ):
                raise ValueError("local qualification lost an assigned root/domain/receiver")


def qualification_cells(
    design: LocalQualificationDesign,
    calibrated: LocalCalibration,
    scores: tuple[LocalRootScore, ...],
) -> tuple[LocalQualificationCell, ...]:
    expected = tuple(r for r, role, _, _ in ROOTS if role == "qualification")
    actual = tuple(dict.fromkeys(s.root for s in scores))
    if actual != expected or calibrated.recipe != ObjectIdentity.from_record(
        design.config_id, design
    ):
        raise ValueError("fresh qualification census or frozen calibration differs")
    cells = []
    for cell in calibrated.cells:
        selected = tuple(
            s for s in scores if s.domain == cell.domain and s.receiver == cell.receiver
        )
        if tuple(s.root for s in selected) != expected:
            raise ValueError("missing assigned local qualification cell")
        contact = tuple(s for s in selected if s.covered_rows and s.action_assay)
        width = (
            None
            if cell.q is None
            else cell.q * design.absolute_scale[cell.receiver]
            + design.numerical_padding[cell.receiver]
        )
        adequate = tuple(
            s.root
            for s in contact
            if not s.invalidity
            and width is not None
            and s.absolute_error is not None
            and s.absolute_error <= width
            and s.response_error is not None
            and s.response_error
            <= design.response_scale[cell.receiver] + 2 * design.numerical_padding[cell.receiver]
        )
        lower = (
            D(0)
            if not adequate
            else D(repr(float(beta.ppf(0.05, len(adequate), len(contact) - len(adequate) + 1))))
        )
        reasons = set(cell.reasons) | {r for s in selected for r in s.invalidity}
        if len(contact) < design.minimum_contact_roots:
            reasons.add("INSUFFICIENT_CAUSAL_DOMAIN_CONTACT")
        if lower < design.qualification_probability_floor:
            reasons.add("LOCAL_HELDOUT_ADEQUACY_FAILED")
        cells.append(
            LocalQualificationCell(
                cell.domain,
                cell.receiver,
                tuple(s.root for s in contact),
                adequate,
                lower,
                width,
                tuple(sorted(reasons)),
            )
        )
    return tuple(cells)
