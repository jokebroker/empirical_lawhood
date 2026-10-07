"""Native assay segments consumed only through the issued campaign runtime provider.

This module owns numerical delivery and bounded in-memory payloads. It has no
filesystem, scheduler, source authority, assay selector or scientific finalizer.
"""

from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256
import io
import json
from typing import Callable

import h5py  # type: ignore[import-untyped]
import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity

from .contracts import BetaCouplingRule, SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView, MatrixIntegratorKind, MatrixPrecision
from .gradients import SixMatrixParameters, coupling_derivative_density
from .model import ComplexArray, SixMatrixState, ideal_state
from .passive_probe import derive_probe_roster
from .response_assay import ResponseGeometryNativeForcePulse, force_step, parent_schedule, select_past_mode
from .response_checkpoint import ResponseGeometryNativeCheckpoint
from .response_observer import ResponseGeometryNativeGeometrySample, ResponseGeometryNativePassiveObserver, PassiveReadout, _decode, _encode, geometry_sample
from .response_qualification import ResponseGeometryAssayNativeConfig, ResponseGeometryAssayNativeDelivery, ResponseGeometryAssayNativeRoot, ResponseGeometryAssayNativeSegmentResult, ResponseGeometryAssayNativeSegment, ResponseGeometryAssayNativeViewSegment
from .shooting import brownian_bridge_split, traceless_hermitian_basis
from .simulation import BAOABGradientCache, SixMatrixResponseRNGStreamReceipt, baoab_step_with_hermitian_noise, checkpoint_from_phase_state, derive_rng_stream, hermitian_noise, state_from_checkpoint


ASSAY_HDF5_SCHEMA = 'empirical-lawhood/simulators/six-matrix-response/assay-native-observations-hdf5'
ASSAY_HDF5_MEDIA_TYPE = "application/x-hdf5"
ASSAY_NATIVE_HDF5_METADATA = {
    "empirical_lawhood_payload_schema": ASSAY_HDF5_SCHEMA,
    "empirical_lawhood_units": '{"couplings":"native-reduced","geometry":"dimensionless","momenta":"native-reduced","positions":"native-reduced","probe":"mixed:reference-ticks,dimensionless,flags","ticks":"reference-tick","transfer":"dimensionless"}',
    "empirical_lawhood_frames": '{"positions":"factor,spatial,matrix-row,matrix-column;unitary-conjugated-native","transfer":"traceless-Hermitian-receiver-basis"}',
    "empirical_lawhood_clocks": '{"probe":"origin-tick,evidence-ready-tick","ticks":"reference-tick=native-step/refinement;dt=0.001"}',
    "empirical_lawhood_keys": '["segment_sha256","refinement","ticks"]',
}


def response_hdf5_text(parent: h5py.Group, name: str, value: str) -> None:
    """Fixed, bounded UTF-8 bytes; no HDF5 variable-length heap references."""
    encoded = value.encode("utf-8")
    if not encoded or len(encoded) > 4096 or b"\0" in encoded:
        raise ValueError("assay HDF5 text is outside its scalar byte bound")
    parent.attrs.create(name, encoded, dtype=f"S{len(encoded)}")


def response_hdf5_read_text(parent: h5py.Group, name: str) -> str:
    if name not in parent.attrs:
        raise ValueError("assay HDF5 identity attribute is absent")
    attribute = parent.attrs.get_id(name)
    try:
        data_type = attribute.get_type()
        if (
            data_type.get_class() != h5py.h5t.STRING
            or data_type.is_variable_str()
            or not 0 < data_type.get_size() <= 4096
            or attribute.get_space().get_simple_extent_ndims() != 0
        ):
            raise ValueError("assay HDF5 identity requires bounded fixed UTF-8 bytes")
    finally:
        attribute.close()
    value = parent.attrs.get(name)
    if isinstance(value, bytes) and 0 < len(value) <= 4096:
        return value.decode("utf-8")
    raise ValueError("assay HDF5 identity requires bounded fixed UTF-8 bytes")


def response_hdf5_writer(stream: io.BytesIO) -> h5py.File:
    """In-memory assay artifact with no wall-clock metadata in scientific bytes."""
    creation = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    creation.set_obj_track_times(False)
    access = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    access.set_fileobj_driver(h5py.h5fd.fileobj_driver, stream)
    return h5py.File(h5py.h5f.create(b"response-geometry-assay-memory", fapl=access, fcpl=creation))


def response_hdf5_group(parent: h5py.Group, name: str) -> h5py.Group:
    creation = h5py.h5p.create(h5py.h5p.GROUP_CREATE)
    creation.set_obj_track_times(False)
    return h5py.Group(h5py.h5g.create(parent.id, name.encode("ascii"), gcpl=creation))


MEMBER = SixMatrixResponseModelFamilyMember(
    "response-geometry.member",
    Decimal("0.5"),
    Decimal("0.5"),
    Decimal(1),
    BetaCouplingRule.PUBLISHED_DETERMINISTIC,
)


def _rng(
    config: ResponseGeometryAssayNativeConfig,
    root: ResponseGeometryAssayNativeRoot,
    purpose: str,
    counter: int = 0,
) -> tuple[np.random.Generator, SixMatrixResponseRNGStreamReceipt]:
    if root not in config.roots or purpose not in {
        "assay.preparation",
        "assay.continuation",
        "assay.bridge",
        "assay.quarter-bridge",
        "assay.covariance",
        "assay.probes",
    }:
        raise ValueError("assay RNG purpose/root is outside the declared roster")
    return derive_rng_stream(
        seed_root_id=f"response-geometry.seed.{config.seed_root_sha256}",
        purpose_id=f"{config.physical_unit_id(root)}.{purpose}",
        stream_index=counter,
        derivation_rule_id="response-geometry.rng.explicit-original-full-seed-pcg64dxsm",
        scientific_seed_sha256=config.scientific_inputs.roots[config.roots.index(root)].native_seed(
            purpose_ordinal=(
                "assay.preparation", "assay.continuation", "assay.bridge", "assay.probes",
                "assay.covariance", "assay.quarter-bridge",
            ).index(purpose),
            counter=counter,
        ),
    )


def _view(refinement: int) -> SixMatrixResponseNumericalView:
    return SixMatrixResponseNumericalView(
        f"response-geometry.view.r{refinement}",
        MatrixIntegratorKind.BAOAB_UNDERDAMPED_LANGEVIN,
        Decimal("0.001") / refinement,
        "dimensionless-langevin-time",
        Decimal(1),
        Decimal(1),
        MatrixPrecision.COMPLEX128,
        "numpy.pcg64dxsm",
        "1.0.0",
        "response-geometry.rng.explicit-original-full-seed-pcg64dxsm",
        512 * refinement,
    )


def _unitary(config: ResponseGeometryAssayNativeConfig, root: ResponseGeometryAssayNativeRoot, refinement: int) -> ComplexArray:
    if not root.covariance or refinement != 2:
        return np.eye(4, dtype="<c16")
    rng, _ = _rng(config, root, "assay.covariance")
    return np.asarray(
        np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0], dtype="<c16"
    )


@dataclass
class _Continuation:
    state: SixMatrixState
    rng: np.random.Generator
    stream: SixMatrixResponseRNGStreamReceipt
    bridge: np.random.Generator
    observer: ResponseGeometryNativePassiveObserver
    window: list[ResponseGeometryNativeGeometrySample]
    x_window: list[ComplexArray]
    mode: ComplexArray | None
    pulse: ResponseGeometryNativeForcePulse | None
    invocation_positions: ComplexArray | None


def _start(
    config: ResponseGeometryAssayNativeConfig,
    segment: ResponseGeometryAssayNativeSegment,
    refinement: int,
    previous: ResponseGeometryNativeCheckpoint | None,
    unitary: ComplexArray,
) -> _Continuation:
    root = segment.root
    if previous is None:
        if segment.phase != "prefix":
            raise ValueError("assay continuation lacks its authoritative predecessor checkpoint")
        coupling = 8.0 if root.context == "prepared" else 0.0
        state = ideal_state(
            q=2,
            alpha_tilde_x=coupling,
            alpha_tilde_y=coupling,
            constitution="11" if root.context == "prepared" else "00",
        )
        state = replace(state, positions=unitary @ state.positions @ unitary.conj().T)
        rng, stream = _rng(config, root, "assay.preparation")
        bridge, _ = _rng(config, root, "assay.bridge")
        _, probe_stream = _rng(config, root, "assay.probes")
        roster = derive_probe_roster(
            config_fingerprint=probe_stream.derived_seed_sha256,
            rule_id="response-geometry-assay.probes",
            scientific_seed=int(config.scientific_inputs.roots[config.roots.index(root)].probe_roster_seed_sha256, 16),
        )
        fields = unitary @ roster.fields @ unitary.conj().T
        coordinates = np.einsum("lab,jab->jl", traceless_hermitian_basis(4).conj(), fields).real
        roster = replace(roster, fields=fields, coordinates=coordinates)
        observer = ResponseGeometryNativePassiveObserver(
            roster=roster, y=state.positions[1], y_velocity=state.momenta[1], refinement=refinement
        )
        return _Continuation(
            state,
            rng,
            stream,
            bridge,
            observer,
            [geometry_sample(state=state, member=MEMBER, refinement=refinement)],
            [state.positions[0]],
            None,
            None,
            None,
        )
    if (
        previous.native.step_index != segment.start_tick * refinement
        or previous.passive.refinement != refinement
        or previous.pending_noises_base64
    ):
        raise ValueError("assay segment must restore the exact completed reference-tick boundary")
    state, rng = state_from_checkpoint(previous.native)
    _, stream = _rng(config, root, "assay.continuation")
    if stream != previous.native.rng_stream:
        raise ValueError("assay predecessor RNG role differs from its declared continuation")
    bridge, _ = _rng(config, root, "assay.bridge")
    bridge.bit_generator.state = json.loads(previous.bridge_rng_state_json)
    value = _Continuation(
        state,
        rng,
        stream,
        bridge,
        ResponseGeometryNativePassiveObserver.restore(previous.passive),
        list(previous.structural_window),
        list(_decode(previous.x_window_base64, (len(previous.structural_window), 3, 4, 4))),
        None
        if previous.frozen_mode_base64 is None
        else _decode(previous.frozen_mode_base64, (3, 4, 4)),
        previous.pulse,
        None
        if previous.invocation_positions_base64 is None
        else _decode(previous.invocation_positions_base64, (2, 3, 4, 4)),
    )
    if segment.phase == "inner":
        assert segment.sign is not None
        value.pulse = ResponseGeometryNativeForcePulse(
            f"action.{segment.task_id}", root.assay, segment.sign, segment.start_tick
        )
        value.invocation_positions = state.positions
    elif segment.phase == "resume":
        assert value.pulse is not None and segment.predecessor is not None
        assert segment.sign is not None
        expected = ResponseGeometryNativeForcePulse(
            f"action.{segment.predecessor.task_id}",
            root.assay,
            segment.sign,
            root.landmark_tick + root.invocation_offset,
        )
        if value.pulse != expected:
            raise ValueError("assay resumed pulse occurrence differs from its checkpoint")
    return value


def _checkpoint(
    value: _Continuation, segment: ResponseGeometryAssayNativeSegment, refinement: int
) -> ResponseGeometryNativeCheckpoint:
    native = checkpoint_from_phase_state(
        checkpoint_id=f"checkpoint.{segment.task_id}.r{refinement}.s{value.state.step_index}",
        request=ObjectIdentity.from_record(segment.task_id, segment),
        total_steps=(
            segment.root.landmark_tick + segment.root.invocation_offset + segment.root.horizon_ticks
        )
        * refinement,
        state=value.state,
        rng=value.rng,
        rng_stream=value.stream,
        receivers=(),
    )
    return ResponseGeometryNativeCheckpoint(
        native.checkpoint_id,
        native,
        value.observer.checkpoint(),
        tuple(value.window),
        _encode(np.asarray(value.x_window, dtype="<c16")),
        None if value.mode is None else _encode(value.mode),
        segment.parent,
        None if segment.parent is None else segment.root.landmark_tick,
        value.pulse,
        None if value.invocation_positions is None else _encode(value.invocation_positions),
        json.dumps(value.bridge.bit_generator.state, sort_keys=True, separators=(",", ":")),
        (),
    )


def _innovations(
    value: _Continuation,
    config: ResponseGeometryAssayNativeConfig,
    segment: ResponseGeometryAssayNativeSegment,
    refinement: int,
    tick: int,
    unitary: ComplexArray,
) -> tuple[ComplexArray, ...]:
    if tick == 256:
        value.rng, value.stream = _rng(config, segment.root, "assay.continuation")
    coarse = hermitian_noise(rng=value.rng, q=2)
    bridge = hermitian_noise(rng=value.bridge, q=2)
    halves = brownian_bridge_split(
        coarse_noise=coarse, bridge_noise=bridge, half_decay=float(np.exp(-0.0005))
    )
    noises: tuple[ComplexArray, ...] = (coarse,) if refinement == 1 else halves
    if refinement == 4:
        quarter, _ = _rng(config, segment.root, "assay.quarter-bridge", tick)
        noises = tuple(
            noise
            for half in halves
            for noise in brownian_bridge_split(
                coarse_noise=half,
                bridge_noise=hermitian_noise(rng=quarter, q=2),
                half_decay=float(np.exp(-0.00025)),
            )
        )
    return tuple(np.asarray(unitary @ noise @ unitary.conj().T, dtype="<c16") for noise in noises)


def _sample(
    rows: list[tuple[object, ...]], value: _Continuation, readout: PassiveReadout | None
) -> None:
    sample = value.window[-1]
    metrics = [
        float("nan") if field is None else float(field)
        for factor in (sample.x, sample.y)
        for field in (factor.phi, factor.closure_ratio, factor.kernel_ratio)
    ]
    probe = [float("nan")] * 9
    transfer = np.full((15, 15), np.nan)
    if readout is not None:
        probe = [
            readout.origin_tick,
            readout.evidence_ready_tick,
            *[
                float("nan") if field is None else field
                for field in (
                    readout.geometry_loss,
                    readout.radius_loss,
                    readout.loss_ratio,
                    readout.drift_loss,
                )
            ],
            int(readout.known),
            -1 if readout.passes is None else int(readout.passes),
            int(readout.transfer_resolved),
        ]
        transfer = _decode(readout.propagator_base64, (15, 15), real=True)
    rows.append(
        (
            sample.reference_tick,
            value.state.positions,
            value.state.momenta,
            (value.state.alpha_tilde_x, value.state.alpha_tilde_y),
            metrics,
            probe,
            transfer,
        )
    )


def _run_view(
    config: ResponseGeometryAssayNativeConfig,
    segment: ResponseGeometryAssayNativeSegment,
    refinement: int,
    previous: ResponseGeometryAssayNativeViewSegment | None,
    group: h5py.Group,
    *,
    progress: Callable[[int], None] | None = None,
) -> ResponseGeometryAssayNativeViewSegment:
    trace = sha256(f"{segment.fingerprint()}.r{refinement}".encode())
    unentered_delivery = ResponseGeometryAssayNativeDelivery(False, 0, 0, 0, Decimal(0), trace.hexdigest(), None)
    if previous is not None and (previous.disposition != "COMPLETE" or previous.checkpoint is None):
        return ResponseGeometryAssayNativeViewSegment(
            refinement,
            "UNENTERED",
            previous.last_completed_native_step,
            None,
            "PREDECESSOR_INCOMPLETE",
            Decimal(0),
            Decimal(0),
            unentered_delivery,
        )
    if (
        previous is not None
        and previous.checkpoint is not None
        and previous.checkpoint.frozen_mode_base64 is None
    ):
        return ResponseGeometryAssayNativeViewSegment(
            refinement,
            "UNENTERED",
            previous.last_completed_native_step,
            None,
            "PAST_MODE_UNRESOLVED",
            Decimal(0),
            Decimal(0),
            unentered_delivery,
        )
    unitary = _unitary(config, segment.root, refinement)
    group.create_dataset("unitary", data=unitary, track_times=False)
    value = _start(
        config, segment, refinement, None if previous is None else previous.checkpoint, unitary
    )
    view, cache = _view(refinement), BAOABGradientCache()
    schedule = (
        None
        if segment.parent is None
        else parent_schedule(parent=segment.parent, refinement=refinement)
    )
    rows: list[tuple[object, ...]] = []
    _sample(rows, value, None)
    work, density_work = 0.0, 0.0
    completed_intervals, nonzero_intervals = 0, 0
    mode_sha256 = None if value.mode is None else sha256(value.mode.tobytes()).hexdigest()
    disposition, reason = "COMPLETE", None
    checkpoint = None
    for tick in range(segment.start_tick, segment.end_tick):
        noises = _innovations(value, config, segment, refinement, tick, unitary)
        for noise in noises:
            old = value.state
            force = 0.0
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    if value.pulse is not None:
                        assert value.mode is not None
                        state = force_step(
                            old,
                            pulse=value.pulse,
                            mode=value.mode,
                            member=MEMBER,
                            numerical_view=view,
                            standardized_noise=noise,
                            gradient_cache=cache,
                        )
                        force = value.pulse.interval_force(
                            native_step=old.step_index, refinement=refinement
                        )
                        next_work = work + force * float(
                            np.vdot(value.mode, state.positions[0] - old.positions[0]).real
                        )
                        next_density_work = density_work
                    else:
                        next_work, next_density_work = work, density_work
                        if segment.phase == "parent":
                            assert schedule is not None
                            offset = old.step_index - segment.start_tick * refinement
                            # The declared parent pulse ends at +384. development's later
                            # invocation slots wait at its returned couplings.
                            alpha = schedule[min(offset, len(schedule) - 1)]
                        else:
                            fraction = min(1.0, (old.step_index + 1) / (256 * refinement))
                            initial = 8.0 if segment.root.context == "prepared" else 0.0
                            alpha = np.asarray(
                                (
                                    initial + fraction * (2 / 3 - initial),
                                    initial + fraction * (22 / 3 - initial),
                                )
                            )
                        state = baoab_step_with_hermitian_noise(
                            old,
                            member=MEMBER,
                            numerical_view=view,
                            next_alpha_tilde_x=float(alpha[0]),
                            next_alpha_tilde_y=float(alpha[1]),
                            standardized_noise=noise,
                            gradient_cache=cache,
                        )
                        if segment.phase == "parent":
                            parameters = SixMatrixParameters(
                                2, 0.5, 0.5, 1.0, old.alpha_tilde_x, old.alpha_tilde_y
                            )
                            derivatives = coupling_derivative_density(old.positions, parameters)
                            next_density_work += abs(
                                derivatives[0] * (state.alpha_tilde_x - old.alpha_tilde_x)
                            )
                            next_density_work += abs(
                                derivatives[1] * (state.alpha_tilde_y - old.alpha_tilde_y)
                            )
                    if not state.finite or not np.isfinite((next_work, next_density_work)).all():
                        raise FloatingPointError("nonfinite native state")
                    value.state = state
                    work, density_work = next_work, next_density_work
                    completed_intervals += 1
                    nonzero_intervals += int(force != 0)
                    trace.update(
                        np.asarray(
                            (
                                old.step_index,
                                old.alpha_tilde_x,
                                old.alpha_tilde_y,
                                state.alpha_tilde_x,
                                state.alpha_tilde_y,
                                force,
                                force,
                            ),
                            dtype="<f8",
                        ).tobytes()
                    )
            except (FloatingPointError, np.linalg.LinAlgError):
                disposition, reason = "NUMERICAL_FAILURE", "NATIVE_NUMERICAL_FAILURE"
                break
            # Bound journal/IPC overhead; each view also reports its final count.
            if progress is not None and completed_intervals % 1024 == 0:
                progress(completed_intervals)
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    readout = value.observer.advance(
                        y=state.positions[1],
                        native_step=state.step_index,
                        y_velocity=state.momenta[1],
                    )
                    if state.step_index % (16 * refinement) == 0:
                        value.window.append(
                            geometry_sample(state=state, member=MEMBER, refinement=refinement)
                        )
                        value.x_window.append(state.positions[0])
                        value.window, value.x_window = value.window[-16:], value.x_window[-16:]
                        _sample(rows, value, readout)
            except (FloatingPointError, np.linalg.LinAlgError):
                disposition, reason = "OBSERVATION_FAILURE", "NATIVE_OBSERVER_FAILURE"
                break
        if reason is not None:
            break
        if (tick + 1) % 512 == 0 and tick + 1 != segment.end_tick:
            intermediate = _checkpoint(value, segment, refinement)
            group.create_dataset(
                f"checkpoint-{tick + 1}",
                data=np.frombuffer(intermediate.canonical_bytes(), dtype="u1"),
                track_times=False,
            )
    if disposition == "COMPLETE":
        if segment.phase == "prefix":
            value.mode = select_past_mode(
                ticks=tuple(sample.reference_tick for sample in value.window),
                observations=np.asarray(value.x_window),
                cutoff_tick=segment.end_tick,
            )
            if value.mode is not None:
                group.create_dataset("shadow_mode", data=value.mode, track_times=False)
        checkpoint = _checkpoint(value, segment, refinement)
    for index, name in enumerate(
        ("ticks", "positions", "momenta", "couplings", "geometry", "probe", "transfer")
    ):
        group.create_dataset(name, data=np.asarray([row[index] for row in rows]), track_times=False)
    return ResponseGeometryAssayNativeViewSegment(
        refinement,
        disposition,
        value.state.step_index,
        checkpoint,
        reason,
        Decimal(str(work)),
        Decimal(str(density_work)),
        ResponseGeometryAssayNativeDelivery(
            True,
            completed_intervals,
            2 * completed_intervals,
            nonzero_intervals,
            Decimal(
                nonzero_intervals
                * (0 if value.pulse is None else value.pulse.sign * int(value.pulse.amplitude))
            )
            * Decimal("0.001")
            / refinement,
            trace.hexdigest(),
            mode_sha256,
        ),
    )


def _execute_response_segment(
    config: ResponseGeometryAssayNativeConfig,
    segment: ResponseGeometryAssayNativeSegment,
    predecessor: ResponseGeometryAssayNativeSegmentResult | None,
    *,
    progress: Callable[[int], None] | None = None,
) -> tuple[ResponseGeometryAssayNativeSegmentResult, bytes]:
    """Execute exactly the planned segment; the production caller owns authority."""
    config_identity = ObjectIdentity.from_record(config.config_id, config)
    if segment.root not in config.roots:
        raise ValueError("assay segment root differs from the issued source roster")
    if segment.predecessor is None:
        if predecessor is not None:
            raise ValueError("assay root prefix cannot consume a predecessor")
    elif (
        predecessor is None
        or predecessor.segment != segment.predecessor
        or predecessor.source_config != config_identity
    ):
        raise ValueError("assay source predecessor/config differs from the exact issued continuation")
    completed_before_view, last_reported = 0, 0

    def report(local_completed: int) -> None:
        nonlocal last_reported
        completed = completed_before_view + local_completed
        if progress is not None and completed > last_reported:
            progress(completed)
            last_reported = completed

    stream = io.BytesIO()
    with response_hdf5_writer(stream) as artifact:
        for name, text in ASSAY_NATIVE_HDF5_METADATA.items():
            if name == "empirical_lawhood_payload_schema":
                text = config.observation_schema
            response_hdf5_text(artifact, name, text)
        response_hdf5_text(artifact, "schema", config.observation_schema)
        response_hdf5_text(artifact, "segment_sha256", segment.fingerprint())
        response_hdf5_text(artifact, "source_config_sha256", config.fingerprint())
        view_results = []
        for index, refinement in enumerate(segment.root.refinements):
            view_result = _run_view(
                config,
                segment,
                refinement,
                None if predecessor is None else predecessor.views[index],
                response_hdf5_group(artifact, f"r{refinement}"),
                progress=report if progress is not None else None,
            )
            report(view_result.delivery.completed_intervals)
            completed_before_view += view_result.delivery.completed_intervals
            view_results.append(view_result)
        views = tuple(view_results)
        if segment.phase == "prefix":
            # The primary mode defines the physical action in every view.
            # Shadow eigendirections are retained only as numerical diagnostics.
            primary = views[0].checkpoint
            primary_mode = (
                None
                if primary is None or primary.frozen_mode_base64 is None
                else _decode(primary.frozen_mode_base64, (3, 4, 4))
            )
            bound_views = []
            for item in views:
                if item.checkpoint is not None:
                    unitary = _unitary(config, segment.root, item.refinement)
                    mode = (
                        None
                        if primary_mode is None
                        else _encode(
                            np.asarray(unitary @ primary_mode @ unitary.conj().T, dtype="<c16")
                        )
                    )
                    item = replace(
                        item, checkpoint=replace(item.checkpoint, frozen_mode_base64=mode)
                    )
                bound_views.append(item)
            views = tuple(bound_views)
    payload = stream.getvalue()
    if len(payload) > 16 * 1024**2:
        raise ValueError("assay native observation shard exceeds its 16-MiB bound")
    result = config.result_type(
        f"result.{segment.task_id}",
        config_identity,
        segment,
        None
        if predecessor is None
        else ObjectIdentity.from_record(predecessor.result_id, predecessor),
        views,
        sha256(payload).hexdigest(),
    )
    return result, payload


def execute_assay_segment(
    config: ResponseGeometryAssayNativeConfig,
    segment: ResponseGeometryAssayNativeSegment,
    predecessor: ResponseGeometryAssayNativeSegmentResult | None,
    *,
    progress: Callable[[int], None] | None = None,
) -> tuple[ResponseGeometryAssayNativeSegmentResult, bytes]:
    """Preserve the strict assay entrypoint over the common native mechanics."""
    if type(config) is not ResponseGeometryAssayNativeConfig or type(segment) is not ResponseGeometryAssayNativeSegment:
        raise ValueError("assay execution requires its exact assay config and segment types")
    return _execute_response_segment(config, segment, predecessor, progress=progress)
