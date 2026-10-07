"Complete fresh local law operands for the existing response-law qualification owner."

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.qualification import cp
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .config import roots
from .discovery import ClassicalNomination, ClassicalRecipe
from .measurement import ClassicalMeasuredRoot
from .prediction import ClassicalInducedPrediction, ClassicalPredictionSeal, predicted_event


@dataclass(frozen=True, slots=True)
class ClassicalQualificationRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-qualification-row'
    block: str
    recipe: ClassicalRecipe
    entered: bool
    roots: tuple[tuple[str, bool, bool, bool, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        if (
            tuple(r[0] for r in self.roots) != roots(self.block, "qualification")
            or any(
                passed and (not known or unsafe or reasons)
                for _, known, passed, unsafe, reasons in self.roots
            )
            or (not self.entered and any(r[2] for r in self.roots))
        ):
            raise ValueError("Local law operands change the complete fixed root/claim denominator")

    @property
    def record_id(self) -> str:
        return f"reactor-staged-pulse-response.{self.block}.{self.recipe.recipe_id}.qualification"

    @property
    def successes(self) -> int:
        return sum(r[2] for r in self.roots)

    @property
    def evaluable(self) -> bool:
        return self.entered and all(r[1] for r in self.roots)

    @property
    def lower(self) -> D:
        return cp(
            self.successes, 96, D(".05") / {"base-menu-comparison": 72, "expanded-menu-comparison": 36, "staged-sequence-comparison": 5}[self.block], lower=True
        )

    @property
    def qualifies(self) -> bool:
        return (
            self.evaluable
            and self.recipe.entered
            and self.successes >= 95
            and self.lower >= D(".90")
            and not any(r[3] for r in self.roots)
        )


@dataclass(frozen=True, slots=True)
class ClassicalQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-qualification'
    nomination: ClassicalNomination
    measurements: tuple[ObjectIdentity, ...]
    assays: tuple[ObjectIdentity, ...]
    predictions: tuple[ObjectIdentity, ...]
    induced_predictions: tuple[ObjectIdentity, ...]
    rows: tuple[ClassicalQualificationRow, ...]

    def __post_init__(self) -> None:
        if (
            len(self.measurements) != 96
            or len(self.assays) != 96
            or len(self.predictions) != 96
            or tuple(r.recipe for r in self.rows) != self.nomination.recipes
            or any(r.block != self.nomination.block for r in self.rows)
            or (self.nomination.block != "staged-sequence-comparison" and self.induced_predictions)
        ):
            raise ValueError("Local law lost its exact nomination, assigned census or prediction cutoffs")

    @property
    def record_id(self) -> str:
        return f"reactor-staged-pulse-response.{self.nomination.block}.qualification"


def qualification_operands(
    nomination: ClassicalNomination,
    measurements: tuple[ClassicalMeasuredRoot, ...],
    predictions: tuple[ClassicalPredictionSeal, ...],
    induced: tuple[ClassicalInducedPrediction, ...] = (),
) -> ClassicalQualification:
    assigned = roots(nomination.block, "qualification")
    parent = ObjectIdentity.from_record(nomination.record_id, nomination)
    if (
        tuple(m.root for m in measurements) != assigned
        or tuple(s.root for s in predictions) != assigned
        or any(
            s.nomination != parent or s.preparation != m.preparation
            for s, m in zip(predictions, measurements, strict=True)
        )
        or len({s.context.root for s in induced}) != len(induced)
    ):
        raise ValueError("Local law changed its full native/prediction census or frozen recipe")
    second = {s.context.root: s for s in induced}
    for s in induced:
        original = next(p for p in predictions if p.root == s.context.root)
        if s.first_seal != ObjectIdentity.from_record(original.record_id, original):
            raise ValueError("second prediction replaces its pre-t0 seal")
    rows = []
    for recipe in nomination.recipes:
        events: list[tuple[str, bool, bool, bool, tuple[str, ...]]] = []
        entered = nomination.entered and recipe.entered
        for measured, seal in zip(measurements, predictions, strict=True):
            row = next(o for o in measured.observations if o.coordinate == recipe.bound.coordinate)
            forecasts = (
                seal.predictions
                if recipe.bound.coordinate.kind != "joint"
                else second[measured.root].predictions
                if measured.root in second
                else ()
            )
            forecast = next((p for p in forecasts if p.recipe_id == recipe.recipe_id), None)
            if not entered:
                events.append((measured.root, False, False, row.unsafe, ("PREREQUISITE_NONENTRY",)))
            elif forecast is None:
                events.append(
                    (measured.root, False, False, row.unsafe, ("MISSING_CAUSAL_PREDICTION",))
                )
            else:
                passed, reasons = predicted_event(forecast, row)
                events.append((measured.root, row.evaluable, passed, row.unsafe, reasons))
        rows.append(ClassicalQualificationRow(nomination.block, recipe, entered, tuple(events)))
    return ClassicalQualification(
        nomination,
        tuple(ObjectIdentity.from_record(m.record_id, m) for m in measurements),
        tuple(m.assay for m in measurements),
        tuple(ObjectIdentity.from_record(s.record_id, s) for s in predictions),
        tuple(ObjectIdentity.from_record(s.record_id, s) for s in induced),
        tuple(rows),
    )
