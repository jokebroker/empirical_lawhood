"""development measurement operands over authenticated native receipts; no fitting or verdict.

assay's packet type is reused only for the identical finite NEG/HOLD/POS assay.
development owns its source, roster, clocks and projection lineage. The HDF reader, exact
continuation join, causal geometry and assay arithmetic retain their existing owners.
"""

from dataclasses import dataclass
from hashlib import sha256
import io
import json
from typing import ClassVar, cast

import numpy as np
from scipy.linalg import expm

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.adapters.simulators.six_matrix_response.response_checkpoint import ResponseGeometryNativeCheckpoint
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import passive_transfer_step, traceless_operator
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import ResponseGeometryDevelopmentNativeConfig, ResponseGeometryDevelopmentNativeRoot, ResponseGeometryDevelopmentNativeSegmentResult, ResponseGeometryDevelopmentNativeSegment
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import PARENTS
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import response_hdf5_group, response_hdf5_text, response_hdf5_writer
from empirical_lawhood.adapters.simulators.six_matrix_response.shooting import hermitian_basis, traceless_hermitian_basis

from .instruments import factor_diagnostic
from .projection import Array, _decimal, _join, _labels, _load, assay_metrics
from .qualification import ResponseGeometryAssayBoundary, ResponseGeometryAssayPacket, parent_work_failure


DEVELOPMENT_KNOTS = (0, 32, 64, 96, 128, 192, 256, 320)
DEVELOPMENT_DATA_SCHEMA = 'empirical-lawhood/methods/response-geometry-prospective/development-measurements-hdf5'

DEVELOPMENT_DATA_METADATA = {
    "empirical_lawhood_payload_schema": DEVELOPMENT_DATA_SCHEMA,
    "empirical_lawhood_units": '{"phase":"native-reduced","ticks":"reference-tick","response":"native-X-HS-displacement"}',
    "empirical_lawhood_frames": '{"phase":"position-X,Y;momentum-X,Y;spatial;global-Hermitian-basis","transfer":"global-traceless-HS-basis"}',
    "empirical_lawhood_clocks": '{"invocation":"causal-parent-end","features":"native-knots;training-future-only","buffers":"past-through-invocation"}',
    "empirical_lawhood_keys": '["root_id","refinement","parent","sign","native_ticks"]',
}


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentProjectionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-projection-config'
    config_id: str
    source_config: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.source_config.object_schema != ResponseGeometryDevelopmentNativeConfig.SCHEMA:
            raise ValueError("development projection requires its exact native config")


def _segments(root: ResponseGeometryDevelopmentNativeRoot) -> tuple[ResponseGeometryDevelopmentNativeSegment, ...]:
    return (
        ResponseGeometryDevelopmentNativeSegment(root, "prefix", None, None),
        *(
            segment
            for parent in PARENTS
            for segment in (
                ResponseGeometryDevelopmentNativeSegment(root, "parent", parent, None),
                *(ResponseGeometryDevelopmentNativeSegment(root, "inner", parent, sign) for sign in (-1, 0, 1)),
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentViewReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-view-report'
    report_id: str
    projection_config: ObjectIdentity
    root: ResponseGeometryDevelopmentNativeRoot
    refinement: int
    source_results: tuple[ObjectIdentity, ...]
    typed_projector_resolved: bool
    packets: tuple[ResponseGeometryAssayPacket, ...]
    data_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.root) is not ResponseGeometryDevelopmentNativeRoot
            or type(self.refinement) is not int
            or self.refinement not in self.root.refinements
        ):
            raise ValueError("development report changes its source root/view")
        if self.report_id != f"report.{self.root.root_id}.r{self.refinement}":
            raise ValueError("development report identity differs from its root/view")
        if self.projection_config.object_schema != ResponseGeometryDevelopmentProjectionConfig.SCHEMA:
            raise ValueError("development report requires its projection config")
        if type(self.typed_projector_resolved) is not bool:
            raise ValueError("development projector resolution must be explicit")
        if tuple(packet.parent for packet in self.packets) != PARENTS:
            raise ValueError("development report drops or adds a parent")
        expected = tuple(sorted(f"result.{segment.task_id}" for segment in _segments(self.root)))
        if tuple(value.object_id for value in self.source_results) != expected or any(
            value.object_schema != ResponseGeometryDevelopmentNativeSegmentResult.SCHEMA for value in self.source_results
        ):
            raise ValueError("development report lacks its exact 21-segment lineage")
        validate_sha256(self.data_sha256, field_name="data_sha256")


def _hs(fields: Array) -> Array:
    return np.asarray(np.einsum("kab,...ab->...k", hermitian_basis(4).conj(), fields).real)


def invocation_buffers(
    checkpoint: ResponseGeometryNativeCheckpoint, path: dict[str, Array]
) -> dict[str, Array]:
    """Exact causal detector/observer state; RNG and queued noise are not features."""
    tick = checkpoint.native.step_index // checkpoint.passive.refinement
    rows = np.flatnonzero((path["ticks"] >= tick - 240) & (path["ticks"] <= tick))
    if len(rows) != 16 or not np.array_equal(
        path["ticks"][rows], np.arange(tick - 240, tick + 1, 16)
    ):
        raise ValueError("development invocation lost its exact causal 16-sample buffer")
    observer = checkpoint.passive
    values = {
        "relative_ticks": path["ticks"][rows] - tick,
        "geometry": path["geometry"][rows],
        "mature_probe": path["probe"][rows],
        "latest_transfer": path["transfer"][rows[-1]],
        "x_window": _hs(_decode(checkpoint.x_window_base64, (16, 3, 4, 4))),
        "passive_fields": _hs(_decode(observer.probes_base64, (13, 3, 4, 4))),
        "passive_y": _hs(_decode(observer.y_base64, (3, 4, 4))),
        "prefix_maxima": np.asarray(
            [
                float(observer.prefix_hermiticity),
                float(observer.prefix_trace),
                float(observer.prefix_identity_drift),
                float(observer.prefix_norm_increase),
            ]
        ),
    }
    for index, pending in enumerate(observer.pending):
        stem = f"pending_{index}"
        values[f"{stem}_relative_tick"] = np.asarray(
            [pending.origin_native_step / observer.refinement - tick]
        )
        values[f"{stem}_fields"] = _hs(_decode(pending.initial_base64, (12, 3, 4, 4)))
        values[f"{stem}_operator"] = _decode(pending.operator_base64, (15, 15), real=True)
        for name, payload in (("y", pending.y_base64), ("momentum", pending.y_velocity_base64)):
            values[f"{stem}_{name}"] = (
                np.full((3, 16), np.nan) if payload is None else _hs(_decode(payload, (3, 4, 4)))
            )
    for index, transfer in enumerate(observer.transfers):
        values[f"transfer_{index}_relative_tick"] = np.asarray(
            [transfer.origin_native_step / observer.refinement - tick]
        )
        values[f"transfer_{index}_fields"] = _hs(_decode(transfer.fields_base64, (15, 4, 4)))
    return values


def phase_features(
    paths: tuple[dict[str, Array], ...], mode: Array, origin: int, slow: Array | None
) -> dict[str, Array]:
    """Fixed causal feature maps; learned centering/scaling/bases belong to fitting."""
    indices = []
    for path in paths:
        rows = np.searchsorted(path["ticks"], origin + np.asarray(DEVELOPMENT_KNOTS))
        if np.any(rows >= len(path["ticks"])) or not np.array_equal(
            path["ticks"][rows], origin + np.asarray(DEVELOPMENT_KNOTS)
        ):
            raise ValueError("development model states are not on their declared native clocks")
        indices.append(rows)
    positions = np.stack(
        [path["positions"][rows] for path, rows in zip(paths, indices, strict=True)]
    )
    momenta = np.stack([path["momenta"][rows] for path, rows in zip(paths, indices, strict=True)])
    fields = np.concatenate((positions, momenta), axis=2)
    scalar = np.stack(
        (
            np.einsum("jab,stjab->st", mode.conj(), positions[:, :, 0]).real,
            np.einsum("jab,stjab->st", mode.conj(), momenta[:, :, 0]).real,
        ),
        axis=-1,
    )
    raw = _hs(fields).reshape(3, 8, 192)
    typed = np.full_like(raw, np.nan)
    if slow is not None:
        basis = np.concatenate((np.eye(4, dtype="complex128")[None] / 2, slow))
        coefficients = np.einsum("kab,stfjab->stfjk", basis.conj(), fields).real
        projected = np.einsum("stfjk,kab->stfjab", coefficients, basis)
        typed = _hs(projected).reshape(3, 8, 192)
    spectra = np.linalg.eigvalsh(fields).reshape(3, 8, 48)
    clocks = np.broadcast_to(np.asarray(DEVELOPMENT_KNOTS)[None, :, None] * 0.001, (3, 8, 1))
    couplings = np.stack(
        [path["couplings"][rows] for path, rows in zip(paths, indices, strict=True)]
    )
    return {
        "scalar": scalar,
        "raw": raw,
        "typed": typed,
        "state_clock": np.concatenate((scalar, spectra, couplings, clocks), axis=-1),
        "native_ticks": origin + np.asarray(DEVELOPMENT_KNOTS),
    }


def compose_native_transfer(path: dict[str, Array], origin: int, horizon: int) -> Array | None:
    """Compose nonoverlapping native maps in their unchanged global frame."""
    if horizon not in (64, 320):
        raise ValueError("development passive diagnostic has another horizon")
    total = np.eye(15)
    for tick in range(origin + 32, origin + horizon + 1, 32):
        rows = np.flatnonzero(path["ticks"] == tick)
        if len(rows) != 1:
            return None
        index = int(rows[0])
        probe, transfer = path["probe"][index], path["transfer"][index]
        if (
            not (probe[0] == tick - 32 and probe[1] == tick and probe[8] == 1)
            or not np.isfinite(transfer).all()
        ):
            return None
        total = transfer @ total
    return total


def passive_origin_forecasts(y: Array, momentum: Array, refinement: int) -> Array:
    """No-change, frozen-Y and linear-Y forecasts through existing native primitives."""
    if type(refinement) is not int or refinement not in (1, 2):
        raise ValueError("development passive forecast changes its numerical view")
    basis = traceless_hermitian_basis(4)
    operator = traceless_operator(y)
    output = np.full((3, 2, 15, 15), np.nan)
    output[0] = np.eye(15)
    for index, horizon in enumerate((64, 320)):
        output[1, index] = expm(-horizon * 0.001 * operator)
    fields = basis.copy()
    dt = 0.001 / refinement
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for step in range(320 * refinement):
                fields = passive_transfer_step(
                    fields=fields,
                    y_start=np.asarray(y + step * dt * momentum, dtype=np.complex128),
                    y_end=np.asarray(y + (step + 1) * dt * momentum, dtype=np.complex128),
                    timestep=dt,
                )
                if step + 1 in (64 * refinement, 320 * refinement):
                    index = int(step + 1 == 320 * refinement)
                    output[2, index] = np.einsum("kab,jab->kj", basis.conj(), fields).real
    except (FloatingPointError, ValueError):
        pass  # An unavailable forecast never changes a measured native map.
    return output


def project_response_geometry_development_view(
    config: ResponseGeometryDevelopmentProjectionConfig,
    root: ResponseGeometryDevelopmentNativeRoot,
    refinement: int,
    inputs: tuple[tuple[ResponseGeometryDevelopmentNativeSegmentResult, bytes], ...],
) -> tuple[ResponseGeometryDevelopmentViewReport, bytes]:
    expected = _segments(root)
    by_segment = {result.segment: result for result, _ in inputs}
    if (
        type(root) is not ResponseGeometryDevelopmentNativeRoot
        or type(refinement) is not int
        or refinement not in root.refinements
        or len(inputs) != len(expected)
        or set(by_segment) != set(expected)
        or any(
            type(result) is not ResponseGeometryDevelopmentNativeSegmentResult
            or result.source_config != config.source_config
            for result, _ in inputs
        )
    ):
        raise ValueError("development projection source roster/config differs")
    for segment, result in by_segment.items():
        if segment.predecessor is not None:
            previous = by_segment[cast(ResponseGeometryDevelopmentNativeSegment, segment.predecessor)]
            if result.predecessor_result != ObjectIdentity.from_record(
                previous.result_id, previous
            ):
                raise ValueError("development projection predecessor lineage differs")
    traces = {result.segment: _load(result, data, refinement) for result, data in inputs}
    views = {
        segment: next(v for v in result.views if v.refinement == refinement)
        for segment, result in by_segment.items()
    }
    prefix = expected[0]
    checkpoint, prefix_trace = views[prefix].checkpoint, traces[prefix]
    mode = (
        None
        if checkpoint is None or checkpoint.frozen_mode_base64 is None
        else _decode(checkpoint.frozen_mode_base64, (3, 4, 4))
    )
    slow = None
    factor_metadata: dict[str, object] = {"reason": "PREPARENT_UNAVAILABLE"}
    if prefix_trace is not None and views[prefix].disposition == "COMPLETE":
        try:
            factor = factor_diagnostic(prefix_trace.arrays["positions"][-1, 1])
            slow = factor.slow_basis
            factor_metadata = {
                "reason": "GAP_UNRESOLVED" if slow is None else None,
                "boundary_gap": factor.boundary_gap,
                "gap_floor": factor.gap_floor,
                "product_defect": factor.product_defect,
                "adjoint_defect": factor.adjoint_defect,
                "center_dimension": factor.center_dimension,
                "commutant_dimension": factor.commutant_dimension,
                "candidate": factor.candidate,
            }
        except (np.linalg.LinAlgError, FloatingPointError):
            factor_metadata = {"reason": "FACTOR_NUMERICAL_FAILURE"}
    packets = []
    origin = root.landmark_tick + root.invocation_offset
    free = float(8 * (0.064 - np.exp(-0.256) * (1 - np.exp(-0.064))))
    output = io.BytesIO()
    with response_hdf5_writer(output) as artifact:
        for name, value in {
            "schema": DEVELOPMENT_DATA_SCHEMA,
            "root_id": root.root_id,
            "projection_config_sha256": config.fingerprint(),
            **DEVELOPMENT_DATA_METADATA,
        }.items():
            response_hdf5_text(artifact, name, value)
        artifact.attrs["refinement"] = refinement
        response_hdf5_text(
            artifact,
            "preparent_factor",
            json.dumps(factor_metadata, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )
        for parent in PARENTS:
            parent_segment = ResponseGeometryDevelopmentNativeSegment(root, "parent", parent, None)
            branches = tuple(
                (prefix, parent_segment, ResponseGeometryDevelopmentNativeSegment(root, "inner", parent, sign))
                for sign in (-1, 0, 1)
            )
            delivered = all(
                views[s].disposition == "COMPLETE" for branch in branches for s in branch
            )
            reasons = {
                reason
                for branch in branches
                for s in branch
                if (reason := views[s].reason) is not None
            }
            work = views[parent_segment].parent_absolute_density_work
            if (failure := parent_work_failure(work)) is not None:
                reasons.add(failure)
            if not delivered or mode is None:
                if mode is None:
                    reasons.add("PAST_MODE_UNRESOLVED")
                packets.append(
                    ResponseGeometryAssayPacket(
                        parent,
                        delivered,
                        None,
                        None,
                        (),
                        (),
                        None,
                        None,
                        _decimal(free),
                        None,
                        None,
                        None,
                        ("U",) * 5,
                        (),
                        (),
                        work,
                        tuple(sorted(reasons)),
                    )
                )
                continue
            paths = tuple(
                _join(tuple(trace for s in branch if (trace := traces[s]) is not None))
                for branch in branches
            )
            origin_index = int(np.flatnonzero(paths[0]["ticks"] == origin)[0])
            measured = assay_metrics(paths, mode, origin_index, free)
            labels = tuple(_labels(path) for path in paths)
            for failed, reason in (
                (not measured.transfer_known, "PRESERVATION_TRANSFER_UNRESOLVED"),
                (not measured.contact, "DIRECT_RESPONSE_CONTACT_ABSENT"),
                (measured.preservation is False, "PRESERVATION_FAILED"),
            ):
                if failed:
                    reasons.add(reason)
            boundaries = tuple(
                ResponseGeometryAssayBoundary(sign, int(path["ticks"][i]), sequence[i - 1], sequence[i])
                for sign, path, sequence in zip((-1, 0, 1), paths, labels, strict=True)
                for i in range(1, len(sequence))
                if sequence[i - 1] in {"G", "N"}
                and sequence[i] in {"G", "N"}
                and sequence[i - 1] != sequence[i]
            )
            packets.append(
                ResponseGeometryAssayPacket(
                    parent,
                    delivered,
                    measured.contact,
                    measured.preservation,
                    tuple(_decimal(v[0]) for v in measured.responses),
                    tuple(_decimal(v[1]) for v in measured.responses),
                    _decimal(measured.odd),
                    _decimal(measured.even),
                    _decimal(free),
                    _decimal(measured.maximum_x),
                    _decimal(measured.maximum_y),
                    _decimal(measured.maximum_transfer) if measured.transfer_known else None,
                    (
                        labels[0][root.landmark_tick // 16],
                        labels[0][origin_index],
                        *(x[-1] for x in labels),
                    ),
                    boundaries,
                    tuple(views[b[-1]].force_work for b in branches),
                    work,
                    tuple(sorted(reasons)),
                )
            )
            group = response_hdf5_group(artifact, parent)
            arrays = phase_features(paths, mode, origin, slow)
            invocation = views[parent_segment].checkpoint
            assert invocation is not None
            buffers = invocation_buffers(invocation, paths[1])
            layout = [(name, list(value.shape)) for name, value in sorted(buffers.items())]
            response_hdf5_text(group, "buffer_layout", json.dumps(layout, separators=(",", ":")))
            arrays["invocation_buffer"] = np.concatenate(
                [buffers[name].ravel() for name, _ in layout]
            )
            arrays["invocation_clock"] = np.asarray(
                [root.landmark_tick, origin, root.invocation_offset]
            )
            arrays["passive_native"] = np.stack(
                [
                    np.stack(
                        [
                            np.full((15, 15), np.nan)
                            if (v := compose_native_transfer(path, origin, h)) is None
                            else v
                            for h in (64, 320)
                        ]
                    )
                    for path in paths
                ]
            )
            arrays["passive_forecasts"] = passive_origin_forecasts(
                paths[1]["positions"][origin_index, 1],
                paths[1]["momenta"][origin_index, 1],
                refinement,
            )
            directions = np.full((6, 4, 4), np.nan, dtype="complex128")
            directions[:3] = mode
            if slow is not None:
                directions[3:] = slow
            arrays["passive_directions"] = np.einsum(
                "kab,jab->kj",
                traceless_hermitian_basis(4).conj(),
                directions,
            ).real
            for name, array in arrays.items():
                group.create_dataset(name, data=array, track_times=False)
    data = output.getvalue()
    if len(data) > 16 * 1024**2:
        raise ValueError("development measurement artifact exceeds its 16-MiB bound")
    return ResponseGeometryDevelopmentViewReport(
        f"report.{root.root_id}.r{refinement}",
        ObjectIdentity.from_record(config.config_id, config),
        root,
        refinement,
        tuple(
            sorted(
                (ObjectIdentity.from_record(r.result_id, r) for r in by_segment.values()),
                key=lambda value: value.object_id,
            )
        ),
        slow is not None,
        tuple(packets),
        sha256(data).hexdigest(),
    ), data
