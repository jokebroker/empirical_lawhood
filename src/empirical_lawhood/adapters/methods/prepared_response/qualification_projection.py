"Pure source qualification scalar reduction of the complete authenticated native root census."

from decimal import Decimal

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS, PreparedRoot
from empirical_lawhood.adapters.simulators.prepared_response.instruments import prepared_force_components, prepared_receiver
from empirical_lawhood.adapters.simulators.prepared_response.source import PreparedNativePhaseData, bind_prepared_native_handoff
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import state_from_checkpoint

from .projection import project_prepared_future, project_prepared_handoff
from .native_projection import authenticate_prepared_root
from .qualification_records import PreparedResponseSourceQualificationProjectionConfig, PreparedResponseSourceQualificationViewObservation, PreparedResponseSourceQualificationWordObservation


def _decimal(value: float) -> Decimal | None:
    return Decimal(str(float(value))) if np.isfinite(value) else None


def _force_error(value: PreparedNativePhaseData) -> Decimal | None:
    delivery = value.delivery
    word = delivery.word
    if word is None or delivery.phase != "future":
        raise ValueError("source qualification force reduction requires its native future invocation")
    if not delivery.completed_intervals:
        return None
    first = delivery.start_tick * delivery.refinement
    components = prepared_force_components(
        word, native_step=first, invocation_tick=delivery.start_tick, refinement=delivery.refinement
    )
    steps = np.arange(delivery.completed_intervals) + first
    expected = (steps < first + 64 * delivery.refinement)[:, None] * components
    error = max(
        float(np.max(np.abs(value.realized_trace[:, 5:7] - expected))),
        float(np.max(np.abs(value.realized_trace[:, 7:9] - expected))),
    )
    if not np.array_equal(value.realized_trace[:, 0], steps):
        raise ValueError("source qualification force trace changed its exact requested native interval clock")
    return _decimal(error)


def project_prepared_response_source_qualification_view(
    config: PreparedResponseSourceQualificationProjectionConfig,
    root: PreparedRoot,
    refinement: int,
    inputs: tuple[tuple[PreparedNativeTaskResult, bytes], ...],
) -> PreparedResponseSourceQualificationViewObservation:
    """Every assigned path is accounted for; unknown observations remain unknown."""
    if (
        root not in config.native_spec.roots
        or type(refinement) is not int
        or refinement not in (1, 2)
    ):
        raise ValueError("prepared source qualification projection is outside its frozen root/view census")
    authenticated = authenticate_prepared_root(config.native_spec, root, inputs)
    identities = {
        result.invocation.task_id: ObjectIdentity.from_record(result.result_id, result)
        for result, _ in inputs
    }
    data, common = authenticated.data, authenticated.common
    velocity = None
    if common is not None and common.frame is not None:
        primary, _ = state_from_checkpoint(common.checkpoints[0].native)
        components = prepared_receiver(common.frame, primary.momenta[0])
        velocity = tuple(Decimal(str(float(v))) for v in components)
    displacements, work, words = [], [], []
    for parent in PARENTS:
        parent_data = data[f"{root.root_id}.{parent}.parent.native"]
        native_parent = None if parent_data is None else parent_data[refinement - 1]
        handoff = (
            None
            if native_parent is None or native_parent.checkpoint is None
            else bind_prepared_native_handoff(native_parent)
        )
        observed = (
            None
            if handoff is None or common is None
            else project_prepared_handoff(
                common, handoff, instrument_tier=config.instrument_tier
            )
        )
        displacements.append(
            None
            if observed is None
            else tuple(Decimal(str(float(v))) for v in observed.preparent_displacement)
        )
        work.append(
            None if native_parent is None else native_parent.delivery.parent_absolute_density_work
        )
        hold_id = f"{root.root_id}.{parent}.common-response.{config.native_spec.words[0].word_id}.native"
        hold_data = data[hold_id]
        hold = None if hold_data is None else hold_data[refinement - 1]
        for word in config.native_spec.words:
            task_id = f"{root.root_id}.{parent}.common-response.{word.word_id}.native"
            pair = data[task_id]
            native = None if pair is None else pair[refinement - 1]
            outputs: tuple[tuple[Decimal | None, ...], ...] = ((None,) * 7,) * 5
            force_error = None
            complete = False
            if native is not None:
                if common is None or handoff is None:
                    raise ValueError("source qualification future entered without an authenticated parent handoff")
                projected = project_prepared_future(common, handoff, native, matched_hold=hold)
                outputs = tuple(tuple(_decimal(float(v)) for v in row) for row in projected.outputs)
                force_error, complete = _force_error(native), projected.delivery_complete
            words.append(
                PreparedResponseSourceQualificationWordObservation(
                    parent, word, identities[task_id], complete, force_error, outputs
                )
            )
    return PreparedResponseSourceQualificationViewObservation(
        ObjectIdentity.from_record(config.config_id, config),
        ObjectIdentity.from_record(config.native_spec.spec_id, config.native_spec),
        root,
        refinement,
        authenticated.identities,
        None if common is None else ObjectIdentity.from_record(common.common_start_id, common),
        "PREFIX_UNAVAILABLE" if common is None else common.mode_disposition,
        velocity,
        tuple(displacements),
        tuple(work),
        tuple(words),
    )
