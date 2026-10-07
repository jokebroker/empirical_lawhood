"Native signed-pair receiver, preserving future purposes and numerical views.\n\nThis is a measurement binding, not a controller or controller-use verdict. Every differential\nreadout retains both branch result identities and its actual matched HOLD.\n"

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.planning.finite_response_geometry import FiniteResponseCoordinate
from empirical_lawhood.runtime.controller_evaluation_nested import PreparedNativeReadout
from .science import FiniteResponseLawScienceSpec
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedForceWord
from .native_records import FiniteResponseLawNativeViewObservation

QUANTITIES = (
    "response-x",
    "response-y",
    "selected-preservation-1",
    "selected-preservation-2",
    "selected-preservation-3",
    "opposite-preservation-1",
    "opposite-preservation-2",
    "opposite-preservation-3",
)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPairedFutureReadout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-paired-future-readout'
    purpose: str
    refinement: int
    projection: ObjectIdentity
    selected_source: ObjectIdentity
    opposite_source: ObjectIdentity
    matched_hold_source: ObjectIdentity
    values: tuple[Decimal | None, ...]
    source_valid: bool

    def __post_init__(self) -> None:
        if (
            self.purpose not in ("future-1", "future-2")
            or type(self.refinement) is not int
            or self.refinement not in (1, 2)
            or len(self.values) != 8
            or type(self.source_valid) is not bool
            or any(
                v is not None and (type(v) is not Decimal or not v.is_finite()) for v in self.values
            )
            or len({self.selected_source, self.opposite_source, self.matched_hold_source}) != 3
            or len(
                {
                    s.object_schema
                    for s in (self.selected_source, self.opposite_source, self.matched_hold_source)
                }
            )
            != 1
            or self.source_valid
            and any(v is None for v in self.values)
        ):
            raise ValueError("Finite response-law paired readout changes its explicit native acquisition roles")


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPairedNativeAssay(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-paired-native-assay'
    physical_independent_unit_id: str
    parent: str
    selected_word: PreparedForceWord
    readouts: tuple[FiniteResponseLawPairedFutureReadout, ...]

    def __post_init__(self) -> None:
        validate_stable_id(
            self.physical_independent_unit_id, field_name="physical_independent_unit_id"
        )
        validate_stable_id(self.parent, field_name="parent")
        if (
            self.selected_word.sign not in (-1, 1)
            or self.selected_word.magnitude not in (Decimal(8), Decimal(16))
            or self.selected_word.direction_index not in (0, 1)
            or tuple((r.purpose, r.refinement) for r in self.readouts)
            != tuple((p, v) for p in ("future-1", "future-2") for v in (1, 2))
        ):
            raise ValueError("Finite response-law paired assay requires both independent futures and both views")

    @property
    def independent_unit_count(self) -> int:
        return 1


def paired_native_assay(
    views: tuple[FiniteResponseLawNativeViewObservation, ...],
    *,
    parent: str,
    word: PreparedForceWord,
) -> FiniteResponseLawPairedNativeAssay:
    """Bind 16 purpose-labelled readouts per view to the full nine-word census.

    No handoff displacement is added and no force-magnitude normalization is
    applied. An absent delivery remains unusable even if its shadow has values.
    """
    if (
        len(views) != 2
        or tuple(v.refinement for v in views) != (1, 2)
        or views[0].root != views[1].root
        or views[0].config != views[1].config
    ):
        raise ValueError("Finite response-law assay requires two views of one exact root and projection")
    if (
        word.sign not in (-1, 1)
        or word.direction_index not in (0, 1)
        or word.magnitude not in (Decimal(8), Decimal(16))
    ):
        raise ValueError("Finite response-law differential receiver requires a nonzero native signed pair")
    expected = {
        (purpose, magnitude, direction, sign)
        for purpose in ("future-1", "future-2")
        for magnitude in (Decimal(8), Decimal(16))
        for direction in (0, 1)
        for sign in (-1, 1)
    } | {(p, Decimal(0), 0, 0) for p in ("future-1", "future-2")}
    result = []
    for view in views:
        actual = {}
        for observation in view.words:
            task = observation.invocation
            if task.parent != parent or task.word is None:
                raise ValueError("Finite response-law paired assay substitutes its assigned parent or word")
            w = task.word
            key = (task.purpose, w.magnitude, w.direction_index, w.sign)
            if key in actual:
                raise ValueError("Finite response-law paired assay duplicates a native acquisition")
            actual[key] = observation
        if set(actual) != expected:
            raise ValueError("Finite response-law paired assay drops a signed mate, future or actual HOLD")
        for purpose in ("future-1", "future-2"):
            selected = actual[(purpose, word.magnitude, word.direction_index, word.sign)]
            opposite = actual[(purpose, word.magnitude, word.direction_index, -word.sign)]
            hold = actual[(purpose, Decimal(0), 0, 0)]
            if (
                selected.matched_hold_result != hold.native_result
                or opposite.matched_hold_result != hold.native_result
                or hold.matched_hold_result != hold.native_result
            ):
                raise ValueError("Finite response-law signed pair substitutes its same-purpose measured HOLD")
            values = (
                tuple(
                    None if a is None or b is None else Decimal(str((float(a) - float(b)) / 2))
                    for a, b in zip(selected.outputs[:2], opposite.outputs[:2], strict=True)
                )
                + selected.outputs[2:5]
                + opposite.outputs[2:5]
            )
            source_valid = all(
                o.complete and o.maximum_force_error == 0 and all(v is not None for v in o.outputs)
                for o in (selected, opposite, hold)
            )
            result.append(
                FiniteResponseLawPairedFutureReadout(
                    purpose,
                    view.refinement,
                    ObjectIdentity.from_record(view.report_id, view),
                    selected.native_result,
                    opposite.native_result,
                    hold.native_result,
                    values,
                    source_valid,
                )
            )
    return FiniteResponseLawPairedNativeAssay(
        views[0].root.physical_unit_id,
        parent,
        word,
        tuple(sorted(result, key=lambda r: (r.purpose, r.refinement))),
    )


def prepared_paired_readouts(
    assay: FiniteResponseLawPairedNativeAssay,
    coordinates: tuple[FiniteResponseCoordinate, ...],
    *,
    refinement: int,
) -> tuple[PreparedNativeReadout, ...]:
    "Translate measured values only; delivery/reveal authority stays with controller use.\n\n    A missing mate or invalid native source omits that future's readouts, causing\n    the generic task intersection to be unevaluable rather than fabricating a\n    successful delivery. The other future and acquisition census remain intact.\n    If all readouts are absent, the delivery/reveal binding must retain an\n    unresolved output locator. A completed generic outcome cannot contain an\n    empty readout list; this helper does not supply its delivery disposition.\n    "
    by_id = {c.coordinate_id: c for c in coordinates}
    expected = {f"{p}.{q}" for p in ("future-1", "future-2") for q in QUANTITIES}
    if (
        refinement not in (1, 2)
        or type(refinement) is not int
        or set(by_id) != expected
        or len(coordinates) != 16
    ):
        raise ValueError("Finite response-law controller use map requires its exact purpose-labelled response chart")
    nu = tuple(d / 8 for d in FiniteResponseLawScienceSpec().delta)
    result = []
    for readout in assay.readouts:
        if readout.refinement != refinement or not readout.source_valid:
            continue
        for quantity, value, floor in zip(QUANTITIES, readout.values, nu, strict=True):
            if value is not None:
                result.append(
                    PreparedNativeReadout(by_id[f"{readout.purpose}.{quantity}"], value, floor)
                )
    return tuple(sorted(result, key=lambda r: r.coordinate.coordinate_id))
