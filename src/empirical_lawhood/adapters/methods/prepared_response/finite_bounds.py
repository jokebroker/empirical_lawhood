"Translate authenticated lower-law evaluations into existing finite admission bounds.\n\nThis is a pure operand conversion. Publication/qualification, complete-view\nassembly, support, raw preservation/effort gates and action selection retain\ntheir existing owners. The HOLD operand is a model prediction, never a hidden\nmatched-HOLD task outcome.\n"

from dataclasses import replace
from decimal import Decimal

from empirical_lawhood.adapters.methods.law_evaluation import (
    LawEvaluationDisposition, LawEvaluationRequest, LawEvaluationResult,
)
from empirical_lawhood.adapters.simulators.prepared_response.contracts import READOUTS
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.finite_response_geometry import FiniteReadoutKind, FiniteResponseBound, FiniteResponseCoordinate


def prepared_finite_bounds(
    *,
    request: LawEvaluationRequest,
    evaluation: LawEvaluationResult,
    hold_request: LawEvaluationRequest,
    hold_evaluation: LawEvaluationResult,
    coordinates: tuple[FiniteResponseCoordinate, ...],
    receiver_quantity_ids: tuple[str, str],
    handoff_displacement: tuple[Decimal, Decimal],
    numerical_floors: tuple[Decimal, Decimal],
) -> tuple[FiniteResponseBound, ...]:
    """Keep the ten readouts, total calibrated width and causal origin shift.

Requests/results come from the existing law service. The response set's owner
binds them to its qualified payload, planned word and joint calibration.
"""
    if len(set(receiver_quantity_ids)) != 2 or len(handoff_displacement) != 2 or len(numerical_floors) != 2:
        raise ValueError("finite prepared response bounds require both receiver coordinates")
    for value in (*handoff_displacement, *numerical_floors):
        if type(value) is not Decimal or not value.is_finite():
            raise ValueError("finite prepared response translation and numerical floors must be resolved")
    if any(value < 0 for value in numerical_floors):
        raise ValueError("finite prepared response numerical floors cannot shrink uncertainty")
    for query, result in ((request, evaluation), (hold_request, hold_evaluation)):
        if (
            result.request != ObjectIdentity.from_record(query.request_id, query)
            or result.evaluator_implementation != query.evaluator
            or result.denominator_member_id != query.denominator_member_id
            or result.candidate_version_id != query.candidate_version_id
            or result.qualification_view_id != query.qualification_view_id
            or query.action_word is None
            or result.action_word != ObjectIdentity.from_record(query.action_word.word_id, query.action_word)
        ):
            raise ValueError("finite prepared response bounds received a detached law evaluation")
        if result.disposition is not LawEvaluationDisposition.SUPPORTED:
            raise ValueError("unavailable lower predictions cannot construct finite prepared response bounds")
    if replace(hold_request, request_id=request.request_id, action_word=request.action_word) != request:
        raise ValueError("finite prepared response baseline must use the same law and public handoff")
    assert hold_request.action_word is not None
    if any(occurrence.realized.value != 0 for occurrence in hold_request.action_word.occurrences):
        raise ValueError("finite prepared response baseline requires the native zero-input HOLD")
    expected_coordinates = {(quantity, Decimal(tick)) for quantity in receiver_quantity_ids for tick in READOUTS}
    if (
        len(coordinates) != 10
        or len({coordinate.coordinate_id for coordinate in coordinates}) != 10
        or {(coordinate.quantity_id, coordinate.end) for coordinate in coordinates} != expected_coordinates
        or any(coordinate.kind is not FiniteReadoutKind.ENDPOINT or coordinate.start != coordinate.end for coordinate in coordinates)
    ):
        raise ValueError("finite prepared response bounds require the complete ten endpoint coordinates")
    means = {value.value_id: value for value in evaluation.response_values}
    baselines = {value.value_id: value for value in hold_evaluation.response_values}
    widths = {value.value_id: value for value in evaluation.uncertainty_values}
    expected_means = {f"mean.t{tick:03d}.c{channel:02d}" for tick in READOUTS for channel in range(2)}
    if set(means) != expected_means or set(baselines) != expected_means:
        raise ValueError("finite prepared response bounds require both receivers at every readout")
    bounds = []
    for coordinate in coordinates:
        channel = receiver_quantity_ids.index(coordinate.quantity_id)
        suffix = f"t{int(coordinate.end):03d}.c{channel:02d}"
        mean, baseline = means[f"mean.{suffix}"], baselines[f"mean.{suffix}"]
        width = widths.get(f"halfwidth.{suffix}")
        if width is None or width.value < 0:
            raise ValueError("finite prepared response bounds require the calibrated total halfwidth")
        if any(
            (value.quantity_id, value.native_unit, value.native_frame_id, value.clock_id)
            != (coordinate.quantity_id, coordinate.unit, coordinate.frame_id, coordinate.clock_id)
            for value in (mean, baseline, width)
        ):
            raise ValueError("finite prepared response response and target coordinates change unit/frame/clock")
        delta = mean.value - baseline.value
        absolute_mean = baseline.value + delta + handoff_displacement[channel]
        bounds.append(FiniteResponseBound(
            f"bound.{evaluation.result_id}.{coordinate.coordinate_id}",
            evaluation.qualification_view_id, coordinate,
            baseline.value, delta, handoff_displacement[channel],
            absolute_mean - width.value, absolute_mean + width.value,
            numerical_floors[channel],
        ))
    return tuple(sorted(bounds, key=lambda value: value.bound_id))
