"Two frozen native-menu consumers expressed in existing finite geometry.\n\nThis module translates law outputs and requests, never selects an action or\nasserts qualification. The shared admission and controller owners retain those decisions.\n"

from dataclasses import dataclass, replace
from decimal import Decimal as D
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.law_evaluation import (
    LawEvaluationDisposition,
    LawEvaluationResult,
)
from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_contracts import FiniteResponseLawEvaluationRoot
from empirical_lawhood.adapters.simulators.finite_response_law.roster import Q_CLOCK, Q_FRAME
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_decimal,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import AvailabilitySpec, CausalPhase
from empirical_lawhood.planning.finite_response_geometry import FiniteReadoutKind, FiniteResponseBound, FiniteResponseCoordinate, FiniteTargetBox, FiniteTargetInterval, FiniteTaskFunctionalKind, FiniteTaskFunctionalSpec
from empirical_lawhood.runtime.controller_evaluation_nested import PreparedNativeReadout

from .law_binding import HORIZON, PAIRED_RECEIVER, output_quantities
from .paired_assay import QUANTITIES
from .science import FiniteResponseLawScienceSpec, root_seed

SPEC = FiniteResponseLawScienceSpec()

NAMES = QUANTITIES


@dataclass(frozen=True, slots=True)
class FiniteResponseLawConsumerRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-consumer-request'

    request_id: str
    root_id: str
    consumer: int
    direction: int
    lower: D
    science: FiniteResponseLawScienceSpec = SPEC

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")
        validate_stable_id(self.root_id, field_name="root_id")
        if type(self.consumer) is not int or self.consumer not in (0, 1):
            raise ValueError("Finite response-law requires one of its two frozen consumers")
        if type(self.direction) is not int or self.direction not in range(4):
            raise ValueError("Finite response-law request requires one of four signed receiver directions")
        validate_decimal(self.lower, field_name="lower")
        lo, hi = self.science.lower_ranges[self.consumer]
        if not lo <= self.lower <= hi or self.science != SPEC:
            raise ValueError("Finite response-law request changes its independently frozen requirement range")


def evaluation_requests(
    root: FiniteResponseLawEvaluationRoot,
) -> tuple[FiniteResponseLawConsumerRequest, FiniteResponseLawConsumerRequest]:
    """Exactly one independent A/B pair for a fresh evaluation root."""
    if not isinstance(root, FiniteResponseLawEvaluationRoot):
        raise TypeError("Finite response-law evaluation requires a validated evaluation root")
    root_id = root.stage_unit
    rng = np.random.Generator(np.random.PCG64(root_seed(root, "prospective-evaluation-request")))
    directions = rng.integers(0, 4, size=2)
    requests = tuple(
        FiniteResponseLawConsumerRequest(
            f"{root_id}.consumer-{c}.request",
            root_id,
            c,
            int(directions[c]),
            D(format(float(rng.uniform(float(lo), float(hi))), ".17g")),
        )
        for c, (lo, hi) in enumerate(SPEC.lower_ranges)
    )
    return requests[0], requests[1]


def response_coordinates(readout: ObjectIdentity) -> tuple[FiniteResponseCoordinate, ...]:
    """Purpose-explicit paired readouts; futures/views never increase root n."""
    return tuple(
        sorted(
            (
                FiniteResponseCoordinate(
                    f"future-{future}.{name}",
                    q.quantity_id,
                    PAIRED_RECEIVER,
                    readout,
                    FiniteReadoutKind.ENDPOINT,
                    q.native_unit,
                    Q_FRAME,
                    Q_CLOCK,
                    D(4368) + HORIZON.duration,
                    D(4368) + HORIZON.duration,
                )
                for future in (1, 2)
                for name, q in zip(NAMES, output_quantities(), strict=True)
            ),
            key=lambda c: c.coordinate_id,
        )
    )


def consumer_task(
    request: FiniteResponseLawConsumerRequest,
    coordinates: tuple[FiniteResponseCoordinate, ...],
) -> FiniteTaskFunctionalSpec:
    if tuple(c.coordinate_id for c in coordinates) != tuple(
        sorted(f"future-{future}.{name}" for future in (1, 2) for name in NAMES)
    ):
        raise ValueError("Finite response-law consumer requires both complete paired future readouts")
    if not coordinates or coordinates != response_coordinates(coordinates[0].readout):
        raise ValueError("Finite response-law consumer changes native receiver, unit, frame or clock")
    quantities = {q.quantity_id: (j, q) for j, q in enumerate(output_quantities())}
    intervals = []
    axis, positive = request.direction // 2, request.direction % 2 == 0
    for coordinate in coordinates:
        j, _ = quantities[coordinate.quantity_id]
        if j == axis:
            lo, hi = request.lower, SPEC.upper[request.consumer]
            if not positive:
                lo, hi = -hi, -lo
        elif j < 2:
            lo, hi = -SPEC.transverse[request.consumer], SPEC.transverse[request.consumer]
        else:
            # With independently enforced h <= delta this redundant lower face
            # represents an upper-only constraint on a nonnegative quantity.
            lo, hi = -SPEC.delta[j], SPEC.preservation[(j - 2) % 3]
        intervals.append(FiniteTargetInterval(coordinate, lo, hi))
    box = FiniteTargetBox(f"{request.request_id}.box", tuple(intervals))
    return FiniteTaskFunctionalSpec(
        f"{request.request_id}.task",
        FiniteTaskFunctionalKind.ENDPOINT_BOX,
        ObjectIdentity.from_record("flh-science", SPEC),
        ObjectIdentity.from_record(request.request_id, request),
        AvailabilitySpec(Q_CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, D(4096)),
        (box,),
        (box.box_id,),
    )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawBoundOperands:
    bounds: tuple[FiniteResponseBound, ...]
    precision_margin: D


def finite_bound_operands(
    result: LawEvaluationResult,
    coordinates: tuple[FiniteResponseCoordinate, ...],
) -> FiniteResponseLawBoundOperands:
    """Project the exact shared-law result; numerical allowance is applied once."""
    if result.disposition is not LawEvaluationDisposition.SUPPORTED:
        raise ValueError("Unavailable law evaluation cannot manufacture finite bounds")
    outputs = output_quantities()
    means = {v.quantity_id: v for v in result.response_values}
    widths = {
        v.quantity_id: v for v in result.uncertainty_values if v.value_id.startswith("halfwidth.")
    }
    numerical = {
        v.quantity_id: v
        for v in result.uncertainty_values
        if v.value_id.startswith("numerical-allowance.")
    }
    ids = {q.quantity_id for q in outputs}
    if (
        len(result.response_values) != 8
        or len(result.uncertainty_values) != 16
        or set(means) != ids
        or set(widths) != ids
        or set(numerical) != ids
    ):
        raise ValueError("Finite response-law finite bounds require complete signed and preservation operands")
    by_id = {q.quantity_id: j for j, q in enumerate(outputs)}
    for q in outputs:
        for v in (means[q.quantity_id], widths[q.quantity_id], numerical[q.quantity_id]):
            if (v.native_unit, v.native_frame_id, v.clock_id) != (
                q.native_unit,
                q.coordinate_frame,
                q.clock_id,
            ):
                raise ValueError("Finite response-law finite bound changes native unit, frame or clock")
        j = by_id[q.quantity_id]
        if widths[q.quantity_id].value < 0 or numerical[q.quantity_id].value != D(
            format(float(SPEC.delta[j] / 8), ".17g")
        ):
            raise ValueError("Finite response-law finite bound changes frozen uncertainty allowance")
    if not coordinates or coordinates != response_coordinates(coordinates[0].readout):
        raise ValueError("Finite response-law finite bounds require the full purpose-explicit coordinate census")
    bounds = []
    for coordinate in coordinates:
        j = by_id[coordinate.quantity_id]
        mean = means[coordinate.quantity_id].value
        width = widths[coordinate.quantity_id].value
        nu = numerical[coordinate.quantity_id].value
        # Domain intersection for preservation is part of the frozen predictor.
        # Keep the numerical floor separate: max(0, mean-width) - nu admits
        # exactly the upper-only requirement when the precision gate passes.
        lo, hi = mean - width, mean + width
        if j >= 2:
            mean, lo, hi = max(D(0), mean), max(D(0), lo), max(D(0), hi)
        bounds.append(
            FiniteResponseBound(
                f"{result.result_id}.{coordinate.coordinate_id}",
                result.qualification_view_id,
                coordinate,
                D(0),
                mean,
                D(0),
                lo,
                hi,
                nu,
            )
        )
    return FiniteResponseLawBoundOperands(
        tuple(sorted(bounds, key=lambda b: b.bound_id)),
        min(
            SPEC.delta[j] - widths[q.quantity_id].value - numerical[q.quantity_id].value
            for j, q in enumerate(outputs)
        ),
    )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawNativeUnitReadoutMap(CanonicalRecord):
    """Explicit positive scaling for the shared scalar target-margin contract.

    Each output is divided by one of its own native units. This changes no
    number or inequality. The source chart retains the original heterogeneous
    units; the derived chart has distinct quantity IDs and dimensionless units.
    The minimum is a noncompensating conjunction, never an effort or utility.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-native-unit-readout-map'
    native_coordinates: tuple[FiniteResponseCoordinate, ...]

    def __post_init__(self) -> None:
        if not self.native_coordinates or self.native_coordinates != response_coordinates(
            self.native_coordinates[0].readout
        ):
            raise ValueError("Finite response-law readout map requires all native paired coordinates")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record("flh-native-unit-readout-map", self)

    @property
    def coordinates(self) -> tuple[FiniteResponseCoordinate, ...]:
        return tuple(
            replace(
                c, quantity_id=f"{c.quantity_id}.in-native-units", readout=self.identity, unit="1"
            )
            for c in self.native_coordinates
        )

    def task(self, native: FiniteTaskFunctionalSpec) -> FiniteTaskFunctionalSpec:
        mapped = dict(zip(self.native_coordinates, self.coordinates, strict=True))
        if any(
            tuple(i.coordinate for i in box.intervals) != self.native_coordinates
            for box in native.boxes
        ):
            raise ValueError("Finite response-law normalized task changes the complete native chart")
        return replace(
            native,
            boxes=tuple(
                replace(
                    box,
                    intervals=tuple(
                        replace(i, coordinate=mapped[i.coordinate]) for i in box.intervals
                    ),
                )
                for box in native.boxes
            ),
        )

    def bounds(
        self, native: tuple[FiniteResponseBound, ...]
    ) -> tuple[FiniteResponseBound, ...]:
        mapped = dict(zip(self.native_coordinates, self.coordinates, strict=True))
        views = {b.qualification_view_id for b in native}
        if (
            not views
            or len(native) != len(views) * 16
            or {(b.qualification_view_id, b.coordinate) for b in native}
            != {(v, c) for v in views for c in self.native_coordinates}
        ):
            raise ValueError("Finite response-law normalized bounds lose native future/view coordinates")
        return tuple(replace(b, coordinate=mapped[b.coordinate]) for b in native)

    def readouts(
        self, native: tuple[PreparedNativeReadout, ...]
    ) -> tuple[PreparedNativeReadout, ...]:
        """Map measured outputs with the identical positive unit conversion.

        Partial observations stay partial: only the evaluator decides the
        consequence of a missing coordinate, signed mate or future.
        """
        mapped = dict(zip(self.native_coordinates, self.coordinates, strict=True))
        if any(r.coordinate not in mapped for r in native) or len(
            {r.coordinate for r in native}
        ) != len(native):
            raise ValueError("Finite response-law normalized readout substitutes or duplicates a coordinate")
        return tuple(replace(r, coordinate=mapped[r.coordinate]) for r in native)
