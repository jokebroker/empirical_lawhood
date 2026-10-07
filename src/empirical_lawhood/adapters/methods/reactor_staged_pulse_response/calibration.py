"""All 15 declared recipes and all 384 historical assigned uses per recipe."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.discovery import FrontierDevelopment
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.measurement import FrontierMeasuredRoot
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.qualification import cp
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.selection import FrontierPredictionSeal
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .config import ARMS, CALIBRATION_ROOTS, GAMMAS, LOCAL_REQUESTS, Coordinate
from .discovery import ClassicalNomination, physical_service, recipe, retained_bounds
from .measurement import retained_observation
from .prediction import choose, predict, predicted_event, retained_points


@dataclass(frozen=True, slots=True)
class CalibrationUse(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/calibration-use'
    root: str
    request_id: str
    chosen: Coordinate | None
    evaluable: bool
    common_service: bool
    declared_success: bool
    unsafe: bool
    mass_kg: D
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.root not in CALIBRATION_ROOTS
            or self.request_id not in tuple(r.request_id for r in LOCAL_REQUESTS)
            or self.mass_kg < 0
            or (
                self.declared_success
                and (
                    not self.common_service
                    or not self.evaluable
                    or self.unsafe
                    or self.chosen is None
                )
            )
            or (
                self.chosen is None
                and (self.mass_kg or self.common_service or self.declared_success or self.unsafe)
            )
        ):
            raise ValueError("historical calibration event changes its assigned use or claim")


@dataclass(frozen=True, slots=True)
class CalibrationCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/calibration-candidate'
    arm: str
    gamma: D
    uses: tuple[CalibrationUse, ...]

    def __post_init__(self) -> None:
        if (
            self.arm not in ARMS[1:]
            or self.gamma not in GAMMAS
            or tuple((u.root, u.request_id) for u in self.uses)
            != tuple((root, r.request_id) for root in CALIBRATION_ROOTS for r in LOCAL_REQUESTS)
        ):
            raise ValueError("calibration changes its 15-candidate/384-use denominator")

    @property
    def successes(self) -> int:
        return sum(u.declared_success for u in self.uses)

    @property
    def failure_roots(self) -> tuple[str, ...]:
        return tuple(
            root
            for root in CALIBRATION_ROOTS
            if any(
                u.root == root and u.chosen is not None and u.evaluable and not u.declared_success
                for u in self.uses
            )
        )

    @property
    def risk_upper(self) -> D:
        return cp(len(self.failure_roots), 96, D(".05") / 15, lower=False)

    @property
    def eligible(self) -> bool:
        return (
            all(u.evaluable and not u.unsafe for u in self.uses)
            and self.successes >= 192
            and self.risk_upper <= D(".10")
        )

    @property
    def successful_mean_feed(self) -> D | None:
        return (
            sum((u.mass_kg for u in self.uses if u.declared_success), D(0)) / self.successes
            if self.successes
            else None
        )


@dataclass(frozen=True, slots=True)
class ClassicalCalibration(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-calibration'
    development: ObjectIdentity
    measurements: tuple[ObjectIdentity, ...]
    predictions: tuple[ObjectIdentity, ...]
    candidates: tuple[CalibrationCandidate, ...]
    selected: tuple[tuple[str, D | None], ...]

    def __post_init__(self) -> None:
        if (
            self.development.object_schema != FrontierDevelopment.SCHEMA
            or len(self.measurements) != 96
            or len(self.predictions) != 96
            or tuple((c.arm, c.gamma) for c in self.candidates)
            != tuple((a, g) for a in ARMS[1:] for g in GAMMAS)
            or self.selected != select_calibration(self.candidates)
        ):
            raise ValueError("calibration changes authenticated inputs or deterministic selection")

    @property
    def record_id(self) -> str:
        return "reactor-staged-pulse-response.12.calibration"


def select_calibration(
    candidates: tuple[CalibrationCandidate, ...],
) -> tuple[tuple[str, D | None], ...]:
    selected = []
    for arm in ARMS[1:]:
        good = [c for c in candidates if c.arm == arm and c.eligible]
        chosen = (
            min(good, key=lambda c: (-c.successes, c.successful_mean_feed or D(0), -c.gamma))
            if good
            else None
        )
        selected.append((arm, None if chosen is None else chosen.gamma))
    return tuple(selected)


def calibrate(
    development: FrontierDevelopment,
    measurements: tuple[FrontierMeasuredRoot, ...],
    predictions: tuple[FrontierPredictionSeal, ...],
) -> tuple[ClassicalCalibration, ClassicalNomination]:
    if (
        tuple(m.root for m in measurements) != CALIBRATION_ROOTS
        or tuple(p.root for p in predictions) != CALIBRATION_ROOTS
    ):
        raise ValueError("calibration cannot omit or replace a historical independent root")
    bounds = retained_bounds(development)
    causal = {m.root: retained_points(p, m) for m, p in zip(measurements, predictions, strict=True)}
    observations = {
        (m.root, b.coordinate): retained_observation(m, b.coordinate)
        for m in measurements
        for b in bounds
    }
    candidates = []
    for arm in ARMS[1:]:
        for gamma in GAMMAS:
            uses = []
            for root in CALIBRATION_ROOTS:
                forecasts = tuple(
                    predict(
                        recipe(arm, b, gamma),
                        next(p for p in causal[root] if p.context == b.coordinate.context),
                    )
                    for b in bounds
                )
                for request in LOCAL_REQUESTS:
                    chosen = choose(forecasts, request)
                    if chosen is None:
                        # Native/preparation failures remain unknown even if a
                        # candidate consequently refuses. Known no-contact is zero service.
                        evaluable = all(
                            o.evaluable
                            for (r, c), o in observations.items()
                            if r == root
                            and c.context == request.context
                            and c.horizon_s == request.horizon_s
                        )
                        uses.append(
                            CalibrationUse(
                                root,
                                request.request_id,
                                None,
                                evaluable,
                                False,
                                False,
                                False,
                                D(0),
                                ("CAUSAL_REFUSAL",),
                            )
                        )
                        continue
                    row = observations[root, chosen.coordinate]
                    common = physical_service(row, request.required_K, request.budget_kg)
                    own, reasons = predicted_event(chosen, row)
                    uses.append(
                        CalibrationUse(
                            root,
                            request.request_id,
                            chosen.coordinate,
                            row.evaluable,
                            common,
                            common and own,
                            row.unsafe,
                            row.applied_masses_kg[0] if row.applied_masses_kg else D(0),
                            reasons,
                        )
                    )
            candidates.append(CalibrationCandidate(arm, gamma, tuple(uses)))
    calibrated = ClassicalCalibration(
        ObjectIdentity.from_record(development.record_id, development),
        tuple(ObjectIdentity.from_record(f"{m.root}.measurement", m) for m in measurements),
        tuple(ObjectIdentity.from_record(p.record_id, p) for p in predictions),
        tuple(candidates),
        select_calibration(tuple(candidates)),
    )
    factors = dict(calibrated.selected)
    factors["EL_INTERVAL"] = factors["EL_SERVICE"]
    recipes = tuple(recipe(arm, b, factors[arm]) for arm in ARMS for b in bounds)
    entered = tuple(
        (arm, "historical-calibration", "strict-assigned-use")
        for arm, factor in calibrated.selected
        if factor is not None
    )
    nomination = ClassicalNomination(
        "base-menu-comparison",
        (ObjectIdentity.from_record(calibrated.record_id, calibrated), calibrated.development),
        recipes,
        (),
        entered,
        () if entered else ("ALL_CALIBRATION_RECIPES_NONENTRY",),
    )
    return calibrated, nomination
