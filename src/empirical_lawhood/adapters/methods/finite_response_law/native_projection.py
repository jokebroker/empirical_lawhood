"Project exact finite response-law signed words with a same-purpose measured HOLD reference."

from decimal import Decimal
from dataclasses import replace
from typing import cast

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.finite_response_law.contracts import FiniteResponseLawNativeRoot, native_invocations
from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawNativeTaskResult, decode_task_native
from empirical_lawhood.adapters.simulators.finite_response_law.source import FiniteResponseLawNativePhaseData
from empirical_lawhood.adapters.simulators.finite_response_law.fresh_contracts import FiniteResponseLawCalibrationRoot
from empirical_lawhood.adapters.simulators.finite_response_law.instruments import compact_interface
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode
from empirical_lawhood.adapters.simulators.prepared_response.instruments import prepared_force_components
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from empirical_lawhood.adapters.methods.prepared_response.projection import NativeResponseArrays, reduce_native_response_arrays
from .native_records import FiniteResponseLawProjectionConfig, FiniteResponseLawNativeViewObservation, native_method_types, FiniteResponseLawCalibrationViewObservation, FiniteResponseLawCalibrationInterface, FiniteResponseLawCalibrationWordObservation


def _calibration_interface(
    phase: FiniteResponseLawNativePhaseData | None,
    record: FiniteResponseLawNativeTaskResult,
) -> FiniteResponseLawCalibrationInterface | None:
    if phase is None or phase.checkpoint is None or record.frame is None:
        return None
    checkpoint = phase.checkpoint
    interface = compact_interface(
        frame=record.frame,
        ticks=checkpoint.history_ticks,
        positions=_decode(checkpoint.history_positions_base64, (31, 2, 3, 4, 4)),
        momenta=_decode(checkpoint.history_momenta_base64, (31, 2, 3, 4, 4)),
        cutoff_tick=record.invocation.clocks[1],
    )
    from empirical_lawhood.adapters.simulators.finite_response_law.source import FiniteResponseLawEvaluationCheckpoint
    from .evaluation_native_records import FiniteResponseLawEvaluationInterface
    from .assigned_native_records import FiniteResponseLawAssignedCalibrationInterface, FiniteResponseLawAssignedEvaluationInterface
    from empirical_lawhood.adapters.simulators.finite_response_law.source import FiniteResponseLawAssignedCalibrationCheckpoint, FiniteResponseLawAssignedEvaluationCheckpoint

    interface_type = (
        FiniteResponseLawAssignedEvaluationInterface
        if type(checkpoint) is FiniteResponseLawAssignedEvaluationCheckpoint
        else FiniteResponseLawAssignedCalibrationInterface
        if type(checkpoint) is FiniteResponseLawAssignedCalibrationCheckpoint
        else FiniteResponseLawEvaluationInterface
        if type(checkpoint) is FiniteResponseLawEvaluationCheckpoint
        else FiniteResponseLawCalibrationInterface
    )
    return interface_type(
        ObjectIdentity.from_record(checkpoint.checkpoint_id, checkpoint),
        interface.instrument,
        interface.cutoff_tick,
        tuple(Decimal(str(float(v))) for v in interface.values),
    )


def _arrays(value: FiniteResponseLawNativePhaseData) -> NativeResponseArrays:
    return NativeResponseArrays(
        value.ticks,
        value.positions,
        value.transfer,
        value.transfer_known,
        float(value.delivery.signed_force_work),
        float(value.delivery.absolute_force_work),
    )


def project_native_view(
    config: FiniteResponseLawProjectionConfig,
    root: FiniteResponseLawNativeRoot,
    refinement: int,
    inputs: tuple[tuple[FiniteResponseLawNativeTaskResult, bytes], ...],
) -> FiniteResponseLawNativeViewObservation:
    if (
        root not in config.native_spec.roots
        or type(refinement) is not int
        or refinement not in (1, 2)
    ):
        raise ValueError("Finite response-law projection changes its assigned root/view")
    tasks = tuple(t for t in native_invocations(config.native_spec) if t.root == root)
    _, _, word_type, view_type, _ = native_method_types(config.native_spec)
    if tuple(r.invocation for r, _ in inputs) != tasks:
        raise ValueError(
            "Finite response-law projection requires its complete ordered native input census"
        )
    records = {r.invocation.task_id: r for r, _ in inputs}
    identities = {
        key: ObjectIdentity.from_record(r.result_id, r) for key, r in records.items()
    }
    values = {r.invocation.task_id: decode_task_native(r, raw) for r, raw in inputs}
    phases = {
        key: None if pair is None else pair[refinement - 1]
        for key, pair in values.items()
    }
    accounting = []
    priors = {p.segment_id: p for p in config.native_spec.retained_predecessors}
    for task in tasks:
        value = phases[task.task_id]
        record = records[task.task_id]
        accounting.append(
            (
                task.task_id,
                0 if value is None else value.delivery.completed_intervals,
                record.unentered_reason
                if value is None
                else value.delivery.disposition,
            )
        )
        if task.dependency_task_ids:
            if record.predecessors != tuple(
                identities[key] for key in task.dependency_task_ids
            ):
                raise ValueError(
                    "Finite response-law projection changes its actual predecessor result hash"
                )
            previous = phases[task.dependency_task_ids[0]]
            if value is not None:
                if previous is None or previous.checkpoint is None:
                    raise ValueError(
                        "Finite response-law entered phase lacks its completed predecessor"
                    )
                checkpoint = previous.checkpoint
                if (
                    value.delivery.incoming_checkpoint
                    != ObjectIdentity.from_record(checkpoint.checkpoint_id, checkpoint)
                    or not np.array_equal(value.positions[0], previous.positions[-1])
                    or not np.array_equal(value.momenta[0], previous.momenta[-1])
                ):
                    raise ValueError("Finite response-law projection changes its incoming native state")
        elif task.predecessor_segment_id is not None:
            declaration = priors[task.predecessor_segment_id]
            artifact = next(
                a
                for a in declaration.artifacts
                if a.payload_schema == PreparedNativeTaskResult.SCHEMA
            )
            if record.predecessors[0].object_fingerprint != artifact.sha256:
                raise ValueError("Finite response-law projection substitutes a retained native result")
    words = []
    hold_word = next(w for w in config.native_spec.words if w.sign == 0)
    for task in (t for t in tasks if t.phase == "future"):
        hold_id = replace(task, word=hold_word).task_id
        record, hold_record = records[task.task_id], records[hold_id]
        value, hold = phases[task.task_id], phases[hold_id]
        outputs: tuple[Decimal | None, ...] = (None,) * 7
        force_error = None
        complete = False
        if value is not None:
            if record.frame is None:
                raise ValueError(
                    "Finite response-law entered future lacks its authenticated fixed frame"
                )
            d = value.delivery
            complete = d.disposition == "COMPLETE"
            if hold is not None:
                hd = hold.delivery
                if (
                    record.common_start != hold_record.common_start
                    or record.frozen_ports_base64 != hold_record.frozen_ports_base64
                    or d.incoming_checkpoint != hd.incoming_checkpoint
                    or d.innovation_streams != hd.innovation_streams
                    or not np.array_equal(value.positions[0], hold.positions[0])
                    or not np.array_equal(value.momenta[0], hold.momenta[0])
                ):
                    raise ValueError(
                        "Finite response-law paired HOLD changes source, frame, handoff or future innovations"
                    )
                if (
                    complete
                    and hd.disposition == "COMPLETE"
                    and d.innovation_sha256 != hd.innovation_sha256
                ):
                    raise ValueError("Finite response-law paired HOLD differs in realized innovations")
            paired = (
                hold is not None
                and complete
                and hold.delivery.disposition == "COMPLETE"
            )
            assert task.word is not None
            observed, _ = reduce_native_response_arrays(
                frame=record.frame,
                word=task.word,
                handoff_tick=4368,
                readouts=(192,),
                future=_arrays(value),
                paired_hold=_arrays(hold) if paired and hold is not None else None,
            )
            outputs = tuple(
                Decimal(str(float(v))) if np.isfinite(v) else None for v in observed[0]
            )
            if d.completed_intervals:
                first = 4368 * refinement
                steps = np.arange(d.completed_intervals) + first
                components = prepared_force_components(
                    task.word,
                    native_step=first,
                    invocation_tick=4368,
                    refinement=refinement,
                )
                expected = (steps < first + 64 * refinement)[:, None] * components
                if not np.array_equal(value.realized_trace[:, 0], steps):
                    raise ValueError(
                        "Finite response-law force trace shifts the requested native clock"
                    )
                force_error = Decimal(
                    str(
                        max(
                            float(
                                np.max(np.abs(value.realized_trace[:, 5:7] - expected))
                            ),
                            float(
                                np.max(np.abs(value.realized_trace[:, 7:9] - expected))
                            ),
                        )
                    )
                )
        words.append(
            word_type(
                task,
                identities[task.task_id],
                identities[hold_id],
                complete,
                force_error,
                outputs,
            )
        )
    # The only None status above would mean a malformed unentered native result;
    # its strict constructor already excludes that state.
    if any(status is None for _, _, status in accounting):
        raise ValueError("Finite response-law native accounting lacks its terminal reason")
    arguments = (
        ObjectIdentity.from_record(config.config_id, config),
        root,
        refinement,
        tuple(identities[t.task_id] for t in tasks),
        tuple((key, n, str(status)) for key, n, status in accounting),
        tuple(words),
    )
    if issubclass(view_type, FiniteResponseLawCalibrationViewObservation):
        if not isinstance(root, FiniteResponseLawCalibrationRoot):
            raise ValueError("Fresh calibration projection requires a fresh root")
        calibration_words = tuple(
            w for w in words if isinstance(w, FiniteResponseLawCalibrationWordObservation)
        )
        if len(calibration_words) != len(words):
            raise ValueError(
                "Fresh calibration projection requires exact native word records"
            )
        prefix = next(t for t in tasks if t.phase == "prefix")
        parent = next(t for t in tasks if t.phase == "parent")
        phase = phases[parent.task_id]
        from .evaluation_native_records import FiniteResponseLawEvaluationViewObservation
        from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawEvaluationTaskResult
        from .control_delivery import observation_words, observe_native_word, _force

        delivery = {}
        if issubclass(view_type, FiniteResponseLawEvaluationViewObservation):
            observations = []
            for word in observation_words(root.root_id):
                selected = tuple(
                    (cast(FiniteResponseLawEvaluationTaskResult, r), raw)
                    for r, raw in inputs
                    if r.invocation.word == _force(word)
                )
                selected = tuple(
                    sorted(selected, key=lambda value: value[0].invocation.purpose)
                )
                observations.append(
                    observe_native_word(
                        word=word, refinement=refinement, inputs=selected
                    ).observed
                )
            delivery["delivery_observations"] = tuple(
                sorted(observations, key=lambda o: o.expected_occurrence_id)
            )
        return view_type(
            arguments[0],
            root,
            refinement,
            arguments[3],
            arguments[4],
            calibration_words,
            _calibration_interface(phases[prefix.task_id], records[prefix.task_id]),
            _calibration_interface(phase, records[parent.task_id]),
            None if phase is None else phase.delivery.parent_absolute_density_work,
            **delivery,
        )
    return view_type(*arguments)
