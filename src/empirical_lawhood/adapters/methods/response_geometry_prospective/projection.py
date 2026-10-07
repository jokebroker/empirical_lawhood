"""Bounded projection of authenticated assay outputs, without native source contact."""

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import io
from typing import Any
from collections.abc import Mapping

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.six_matrix_response.model import hermiticity_residual
from empirical_lawhood.adapters.simulators.six_matrix_response.response_assay import signed_displacement
from empirical_lawhood.adapters.simulators.six_matrix_response.response_hessian import endogenous_hessian
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import ResponseGeometryNativeFactorMetrics, ResponseGeometryNativeGeometrySample, _decode
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import ResponseGeometryAssayNativeRoot, ResponseGeometryAssayNativeSegmentResult, ResponseGeometryAssayNativeSegment, PARENTS, assay_segments
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import response_hdf5_group, response_hdf5_read_text, response_hdf5_text, response_hdf5_writer
from empirical_lawhood.adapters.simulators.six_matrix_response.shooting import hermitian_basis

from .instruments import causal_geometry, factor_diagnostic, structural_window
from .qualification import ResponseGeometryAssayBoundary, ResponseGeometryAssayPacket, ResponseGeometryAssayProjectionConfig, ResponseGeometryAssayViewReport, parent_work_failure


ASSAY_DIAGNOSTICS_SCHEMA = 'empirical-lawhood/methods/response-geometry-prospective/assay-diagnostics-hdf5'
ASSAY_DIAGNOSTICS_HDF5_METADATA = {
    "empirical_lawhood_payload_schema": ASSAY_DIAGNOSTICS_SCHEMA,
    "empirical_lawhood_units": '{"factor":"dimensionless","hessian":"native-reduced;unit-kinetic-mass","passive_losses":"mixed:reference-ticks,dimensionless,flags","ticks":"reference-tick"}',
    "empirical_lawhood_frames": '{"factor":"ordered-X,Y-factor-diagnostics","hessian":"factor,spatial,Hermitian-basis;96-real-coordinates"}',
    "empirical_lawhood_clocks": '{"passive_losses":"origin-tick,evidence-ready-tick","ticks":"reference-tick=native-step/refinement;dt=0.001"}',
    "empirical_lawhood_keys": '["root_id","refinement","parent","sign","ticks"]',
}
_ARRAYS = ("ticks", "positions", "momenta", "couplings", "geometry", "probe", "transfer")
Array = npt.NDArray[Any]


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("assay projection cannot serialize a nonfinite measured operand")
    return Decimal(str(value))


@dataclass(frozen=True)
class _Trace:
    arrays: dict[str, Array]
    unitary: Array
    shadow_mode: Array | None


def _load(result: ResponseGeometryAssayNativeSegmentResult, payload: bytes, refinement: int) -> _Trace | None:
    if len(payload) > 16 * 1024**2 or sha256(payload).hexdigest() != result.observations_sha256:
        raise ValueError("assay native payload violates its size or authenticated digest")
    with h5py.File(io.BytesIO(payload), "r") as artifact:
        if (
            response_hdf5_read_text(artifact, "schema") != result.OBSERVATION_SCHEMA
            or response_hdf5_read_text(artifact, "segment_sha256") != result.segment.fingerprint()
            or response_hdf5_read_text(artifact, "source_config_sha256")
            != result.source_config.object_fingerprint
        ):
            raise ValueError("assay native payload schema/segment/config differs")
        if tuple(sorted(artifact.keys())) != tuple(
            f"r{value}" for value in result.segment.root.refinements
        ):
            raise ValueError("assay native payload changes its numerical-view roster")
        if any(
            not isinstance(artifact.get(name, getlink=True), h5py.HardLink)
            or not isinstance(artifact[name], h5py.Group)
            for name in artifact
        ):
            raise ValueError("assay native view cannot link to another artifact")
        group = artifact[f"r{refinement}"]
        view = next(value for value in result.views if value.refinement == refinement)
        if view.disposition == "UNENTERED":
            if tuple(group.keys()):
                raise ValueError("assay unentered view contains observations")
            return None
        for name in group:
            if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                raise ValueError("assay native payload cannot contain external or symbolic links")
            dataset = group[name]
            if (
                not isinstance(dataset, h5py.Dataset)
                or dataset.is_virtual
                or dataset.external
                or dataset.id.get_create_plist().get_nfilters() != 0
            ):
                raise ValueError("assay native payload requires local unfiltered numeric datasets")
        maximum_rows = (
            view.last_completed_native_step // refinement - result.segment.start_tick
        ) // 16 + 1
        if "ticks" not in group or len(group["ticks"].shape) != 1:
            raise ValueError("assay native clock array is absent or has another rank")
        rows = int(group["ticks"].shape[0])
        if not 1 <= rows <= maximum_rows or (
            view.disposition == "COMPLETE" and rows != maximum_rows
        ):
            raise ValueError("assay native samples exceed or truncate the delivered clock")
        shapes = (
            (rows,),
            (rows, 2, 3, 4, 4),
            (rows, 2, 3, 4, 4),
            (rows, 2),
            (rows, 6),
            (rows, 9),
            (rows, 15, 15),
        )
        arrays = {}
        dtypes = ("int64", "complex128", "complex128", "float64", "float64", "float64", "float64")
        for name, shape, dtype in zip(_ARRAYS, shapes, dtypes, strict=True):
            if (
                name not in group
                or not isinstance(group.get(name, getlink=True), h5py.HardLink)
                or group[name].shape != shape
                or group[name].dtype != np.dtype(dtype)
            ):
                raise ValueError("assay native array differs from its bounded numeric geometry")
            arrays[name] = np.asarray(group[name][...])
        expected_ticks = np.arange(
            result.segment.start_tick, result.segment.start_tick + 16 * rows, 16
        )
        if not np.array_equal(arrays["ticks"], expected_ticks):
            raise ValueError("assay native samples change the frozen reference-clock lattice")
        for name in ("positions", "momenta", "couplings"):
            if not np.isfinite(arrays[name]).all():
                raise ValueError("assay retained native fields are nonfinite")
        for name in ("positions", "momenta"):
            if (
                arrays[name].dtype != np.dtype("complex128")
                or hermiticity_residual(arrays[name]) > 1e-12
            ):
                raise ValueError("assay retained native fields change Hermitian complex128 geometry")
        if (
            "unitary" not in group
            or group["unitary"].shape != (4, 4)
            or group["unitary"].dtype != np.dtype("complex128")
        ):
            raise ValueError("assay native unitary has another geometry")
        unitary = np.asarray(group["unitary"][...])
        if (
            not np.isfinite(unitary).all()
            or np.linalg.norm(unitary.conj().T @ unitary - np.eye(4)) > 1e-12
        ):
            raise ValueError("assay native gauge map is not unitary")
        shadow = None
        if "shadow_mode" in group:
            if group["shadow_mode"].shape != (3, 4, 4) or group["shadow_mode"].dtype != np.dtype(
                "complex128"
            ):
                raise ValueError("assay shadow mode has another geometry")
            shadow = np.asarray(group["shadow_mode"][...])
            if (
                not np.isfinite(shadow).all()
                or hermiticity_residual(shadow) > 1e-12
                or abs(float(np.vdot(shadow, shadow).real) - 1) > 1e-12
            ):
                raise ValueError("assay shadow mode has unresolved normalization")
        return _Trace(arrays, unitary, shadow)


def _join(traces: tuple[_Trace, ...]) -> dict[str, Array]:
    if not traces:
        raise ValueError("assay path lacks its declared segments")
    for left, right in zip(traces, traces[1:]):
        if left.arrays["ticks"][-1] != right.arrays["ticks"][0] or any(
            not np.array_equal(left.arrays[name][-1], right.arrays[name][0], equal_nan=True)
            for name in ("positions", "momenta", "couplings", "geometry")
        ):
            raise ValueError("assay continuation does not restore its exact native endpoint")
    # Keep the predecessor's mature readout at shared boundaries; the restored
    # task's initial row intentionally has no newly matured observation.
    return {
        name: np.concatenate(
            tuple(
                trace.arrays[name] if index == 0 else trace.arrays[name][1:]
                for index, trace in enumerate(traces)
            )
        )
        for name in _ARRAYS
    }


@dataclass(frozen=True)
class AssayMetrics:
    responses: tuple[tuple[float, float, float], ...]
    odd: float
    even: float
    contact: bool
    maximum_x: float
    maximum_y: float
    maximum_transfer: float
    transfer_known: bool
    preservation: bool | None


def assay_metrics(
    paths: tuple[dict[str, Array], ...], mode: Array, origin_index: int, free: float
) -> AssayMetrics:
    """Shared finite assay arithmetic; no panel selection or qualification."""
    origin = paths[0]["positions"][origin_index]
    responses = tuple(
        signed_displacement(mode=mode, origin=origin[0], current=path["positions"][-1, 0])
        for path in paths
    )
    odd = (responses[2][0] - responses[0][0]) / 2
    even = (responses[2][0] + responses[0][0]) / 2 - responses[1][0]
    contact = abs(odd) >= 1 / 32 and abs(even) <= 1 / 32 and abs(odd - free) >= 1 / 128
    maximum_x, maximum_y, maximum_transfer = 0.0, 0.0, 0.0
    transfer_known = True
    hold = paths[1]
    for action in (paths[0], paths[2]):
        displacement = action["positions"][origin_index:] - hold["positions"][origin_index:]
        projection = np.einsum("jab,sjab->s", mode.conj(), displacement[:, 0]).real
        orthogonal = displacement[:, 0] - projection[:, None, None, None] * mode
        maximum_x = max(
            maximum_x,
            float(np.max(np.linalg.norm(orthogonal.reshape(len(orthogonal), -1), axis=1))),
        )
        maximum_y = max(
            maximum_y,
            float(np.max(np.linalg.norm(displacement[:, 1].reshape(len(displacement), -1), axis=1)))
            / max(1, float(np.linalg.norm(origin[1]))),
        )
        for index in range(origin_index + 2, len(action["ticks"])):
            tick = int(action["ticks"][index])
            first, second = action["probe"][index], hold["probe"][index]
            if (
                not all(
                    row[0] == tick - 32 and row[1] == tick and row[8] == 1
                    for row in (first, second)
                )
                or not np.isfinite(action["transfer"][index]).all()
                or not np.isfinite(hold["transfer"][index]).all()
            ):
                transfer_known = False
                continue
            maximum_transfer = max(
                maximum_transfer,
                float(np.linalg.norm(action["transfer"][index] - hold["transfer"][index], 2)),
            )
    preservation = (
        None
        if not transfer_known
        else maximum_x <= 0.125 and maximum_y <= 0.05 and maximum_transfer <= 0.05
    )
    return AssayMetrics(
        responses,
        odd,
        even,
        bool(contact),
        maximum_x,
        maximum_y,
        maximum_transfer,
        transfer_known,
        preservation,
    )


def _labels(arrays: dict[str, Array]) -> tuple[str, ...]:
    samples: list[ResponseGeometryNativeGeometrySample] = []
    labels = []
    for index, raw_tick in enumerate(arrays["ticks"]):
        tick = int(raw_tick)
        fields = tuple(
            None if not np.isfinite(value) else _decimal(float(value))
            for value in arrays["geometry"][index]
        )
        samples.append(
            ResponseGeometryNativeGeometrySample(
                tick, ResponseGeometryNativeFactorMetrics(*fields[:3]), ResponseGeometryNativeFactorMetrics(*fields[3:])
            )
        )
        samples = samples[-16:]
        x, y, y_strict = structural_window(tuple(samples))
        readout = arrays["probe"][index]
        # Drift normalization may be unknown independently of the geometry
        # predicate; only its own required operands determine availability.
        known = bool(
            np.isfinite(readout[[0, 1, 2, 3, 4, 6, 7]]).all()
            and readout[6] == 1
            and readout[7] in {0, 1}
        )
        observation = causal_geometry(
            x_geometric=x,
            y_geometric=y,
            y_strict_window=y_strict,
            passive_pass=bool(readout[7]) if known else None,
            acquisition_tick=int(readout[0]) if known else tick,
            evidence_ready_tick=int(readout[1]) if known else tick,
            decision_tick=tick,
        )
        labels.append(observation.label)
    return tuple(labels)


def _factor_rows(positions: Array) -> Array:
    """Pure diagnostics on one authenticated segment, including unknown bands."""
    factor_rows: list[tuple[float, ...]] = []
    for state in positions:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                diagnostic = factor_diagnostic(state[1])
        except (FloatingPointError, np.linalg.LinAlgError):
            # An unresolved spectral gap retains finite gap/floor operands;
            # a numerical failure has neither and gets an explicit output mask.
            factor_rows.append((float("nan"),) * 7)
            continue
        factor_rows.append(
            tuple(
                float("nan") if value is None else float(value)
                for value in (
                    diagnostic.boundary_gap,
                    diagnostic.gap_floor,
                    diagnostic.product_defect,
                    diagnostic.adjoint_defect,
                    diagnostic.center_dimension,
                    diagnostic.commutant_dimension,
                    diagnostic.candidate,
                )
            )
        )
    return np.asarray(factor_rows)


class _FactorProjection:
    """One projection-local cache keyed by the already authenticated segment.

    A signed branch reuses its exact prefix/parent factors. Cache entries never
    cross a root/view/task or acquire new native data; every output row remains.
    """

    def __init__(self, traces: Mapping[ResponseGeometryAssayNativeSegment, _Trace | None]):
        self.traces = traces
        self.rows: dict[ResponseGeometryAssayNativeSegment, Array] = {}

    def path(self, segments: tuple[ResponseGeometryAssayNativeSegment, ...]) -> Array:
        values = []
        for index, segment in enumerate(segments):
            if segment not in self.rows:
                trace = self.traces[segment]
                if trace is None:
                    raise ValueError("assay factor projection lacks a declared native segment")
                self.rows[segment] = _factor_rows(trace.arrays["positions"])
            values.append(self.rows[segment] if index == 0 else self.rows[segment][1:])
        return np.concatenate(values)


def _write_diagnostics(
    group: Any,
    arrays: dict[str, Array],
    labels: tuple[str, ...],
    mode: Array | None,
    hessian_ticks: tuple[int, ...],
    factors: Array,
) -> None:
    group.create_dataset("ticks", data=arrays["ticks"], track_times=False)
    group.create_dataset("geometry_labels", data=np.asarray(labels, dtype="S1"), track_times=False)
    group.create_dataset("passive_losses", data=arrays["probe"], track_times=False)
    group.create_dataset("factor", data=factors, track_times=False)
    group.create_dataset(
        "factor_numerically_resolved",
        data=np.isfinite(factors[:, :2]).all(axis=1),
        track_times=False,
    )
    for tick in hessian_ticks:
        indices = np.flatnonzero(arrays["ticks"] == tick)
        if len(indices) != 1:
            continue
        snapshot = response_hdf5_group(group, f"hessian-{tick}")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                hessian = endogenous_hessian(arrays["positions"][indices[0]])
                eigenvalues, eigenvectors = np.linalg.eigh(hessian.matrix)
                discriminant = np.sqrt(np.asarray(1 - 4 * eigenvalues, dtype="complex128"))
        except (FloatingPointError, np.linalg.LinAlgError):
            response_hdf5_text(snapshot, "disposition", "NUMERICAL_FAILURE")
            snapshot.attrs["resolved"] = False
            continue
        response_hdf5_text(snapshot, "disposition", "RESOLVED" if hessian.resolved else "UNRESOLVED")
        snapshot.create_dataset("matrix", data=hessian.matrix, track_times=False)
        snapshot.attrs["refinement_error"] = hessian.refinement_error
        snapshot.attrs["symmetry_error"] = hessian.symmetry_error
        snapshot.attrs["resolved"] = hessian.resolved
        snapshot.create_dataset("eigenvalues", data=eigenvalues, track_times=False)
        if mode is not None:
            force = np.zeros(96)
            force[:48] = np.einsum("kab,jab->jk", hermitian_basis(4).conj(), mode).real.ravel()
            snapshot.create_dataset(
                "input_receiver_overlap", data=np.square(eigenvectors.T @ force), track_times=False
            )
        # Unit kinetic mass and friction one: eigenvalues solve z²+z+lambda=0.
        snapshot.create_dataset(
            "phase_drift_rates",
            data=np.column_stack(((-1 + discriminant) / 2, (-1 - discriminant) / 2)),
            track_times=False,
        )


def project_response_geometry_assay_view(
    config: ResponseGeometryAssayProjectionConfig,
    root: ResponseGeometryAssayNativeRoot,
    refinement: int,
    inputs: tuple[tuple[ResponseGeometryAssayNativeSegmentResult, bytes], ...],
) -> tuple[ResponseGeometryAssayViewReport, bytes]:
    expected = tuple(segment for segment in assay_segments() if segment.root == root)
    by_segment = {result.segment: result for result, _ in inputs}
    if (
        len(by_segment) != len(inputs)
        or set(by_segment) != set(expected)
        or refinement not in root.refinements
        or any(result.source_config != config.source_config for result, _ in inputs)
    ):
        raise ValueError("assay projection source roster/config differs")
    for segment, result in by_segment.items():
        if segment.predecessor is not None:
            previous = by_segment[segment.predecessor]
            if result.predecessor_result != ObjectIdentity.from_record(
                previous.result_id, previous
            ):
                raise ValueError("assay projection predecessor lineage differs")
    traces = {result.segment: _load(result, payload, refinement) for result, payload in inputs}
    factors = _FactorProjection(traces)
    prefix = ResponseGeometryAssayNativeSegment(root, "prefix", None, None)
    prefix_result = next(
        value for value in by_segment[prefix].views if value.refinement == refinement
    )
    prefix_trace = traces[prefix]
    mode = (
        None
        if prefix_result.checkpoint is None or prefix_result.checkpoint.frozen_mode_base64 is None
        else _decode(prefix_result.checkpoint.frozen_mode_base64, (3, 4, 4))
    )
    shadow_distance = None
    if mode is not None and prefix_trace is not None and prefix_trace.shadow_mode is not None:
        overlap = float(np.vdot(mode, prefix_trace.shadow_mode).real)
        shadow_distance = _decimal(float(np.sqrt(max(0, 1 - min(1, overlap * overlap)))))
    output = io.BytesIO()
    packets = []
    with response_hdf5_writer(output) as diagnostics:
        for name, text in ASSAY_DIAGNOSTICS_HDF5_METADATA.items():
            response_hdf5_text(diagnostics, name, text)
        response_hdf5_text(diagnostics, "schema", ASSAY_DIAGNOSTICS_SCHEMA)
        response_hdf5_text(diagnostics, "root_id", root.root_id)
        diagnostics.attrs["refinement"] = refinement
        response_hdf5_text(diagnostics, "projection_config_sha256", config.fingerprint())
        if prefix_trace is not None:
            _write_diagnostics(
                response_hdf5_group(diagnostics, "prefix"),
                prefix_trace.arrays,
                _labels(prefix_trace.arrays),
                mode,
                (root.landmark_tick,),
                factors.path((prefix,)),
            )
        for parent in PARENTS:
            parent_segment = ResponseGeometryAssayNativeSegment(root, "parent", parent, None)
            branches = tuple(
                (
                    prefix,
                    parent_segment,
                    ResponseGeometryAssayNativeSegment(root, "inner", parent, sign),
                    *((ResponseGeometryAssayNativeSegment(root, "resume", parent, sign),) if root.restart else ()),
                )
                for sign in (-1, 0, 1)
            )
            views = {
                segment: next(
                    value for value in by_segment[segment].views if value.refinement == refinement
                )
                for branch in branches
                for segment in branch
            }
            delivered = all(value.disposition == "COMPLETE" for value in views.values())
            reasons = {value.reason for value in views.values() if value.reason is not None}
            parent_work = views[parent_segment].parent_absolute_density_work
            work_failure = parent_work_failure(parent_work)
            if work_failure is not None:
                reasons.add(work_failure)
            free = (8 if root.assay == "short-pulse-response" else 4) * (
                root.pulse_ticks * 0.001
                - np.exp(-(root.horizon_ticks - root.pulse_ticks) * 0.001)
                * (1 - np.exp(-root.pulse_ticks * 0.001))
            )
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
                        _decimal(float(free)),
                        None,
                        None,
                        None,
                        ("U",) * 5,
                        (),
                        (),
                        parent_work,
                        tuple(sorted(reasons)),
                    )
                )
                continue
            paths = tuple(
                _join(tuple(trace for segment in branch if (trace := traces[segment]) is not None))
                for branch in branches
            )
            labels = tuple(_labels(path) for path in paths)
            origin_tick = root.landmark_tick + 384
            origin_index = int(np.flatnonzero(paths[0]["ticks"] == origin_tick)[0])
            measured = assay_metrics(paths, mode, origin_index, float(free))
            responses, odd, even = measured.responses, measured.odd, measured.even
            contact, preservation = measured.contact, measured.preservation
            maximum_x, maximum_y = measured.maximum_x, measured.maximum_y
            maximum_transfer, transfer_known = measured.maximum_transfer, measured.transfer_known
            if not transfer_known:
                reasons.add("PRESERVATION_TRANSFER_UNRESOLVED")
            if not contact:
                reasons.add("DIRECT_RESPONSE_CONTACT_ABSENT")
            if preservation is False:
                reasons.add("PRESERVATION_FAILED")
            boundaries = tuple(
                ResponseGeometryAssayBoundary(sign, int(path["ticks"][index]), sequence[index - 1], sequence[index])
                for sign, path, sequence in zip((-1, 0, 1), paths, labels, strict=True)
                for index in range(1, len(sequence))
                if sequence[index - 1] in {"G", "N"}
                and sequence[index] in {"G", "N"}
                and sequence[index - 1] != sequence[index]
            )
            group = response_hdf5_group(diagnostics, parent)
            for sign, path, sequence, branch in zip(
                (-1, 0, 1), paths, labels, branches, strict=True
            ):
                # One parent-end Hessian; the three inner paths share its state.
                path_factors = factors.path(branch)
                if not np.isfinite(path_factors[:, :2]).all():
                    reasons.add("FACTOR_DIAGNOSTIC_NUMERICAL_FAILURE")
                _write_diagnostics(
                    response_hdf5_group(group, {-1: "neg", 0: "hold", 1: "pos"}[sign]),
                    path,
                    sequence,
                    mode,
                    (origin_tick,) if sign == 0 else (),
                    path_factors,
                )
            packets.append(
                ResponseGeometryAssayPacket(
                    parent,
                    delivered,
                    bool(contact),
                    preservation,
                    tuple(_decimal(value[0]) for value in responses),
                    tuple(_decimal(value[1]) for value in responses),
                    _decimal(odd),
                    _decimal(even),
                    _decimal(float(free)),
                    _decimal(maximum_x),
                    _decimal(maximum_y),
                    _decimal(maximum_transfer) if transfer_known else None,
                    (
                        labels[0][root.landmark_tick // 16],
                        labels[0][origin_index],
                        *(sequence[-1] for sequence in labels),
                    ),
                    boundaries,
                    tuple(
                        sum((views[segment].force_work for segment in branch[2:]), Decimal(0))
                        for branch in branches
                    ),
                    parent_work,
                    tuple(sorted(reasons)),
                )
            )
    payload = output.getvalue()
    if len(payload) > 16 * 1024**2:
        raise ValueError("assay projection diagnostics exceed their 16-MiB bound")
    report = ResponseGeometryAssayViewReport(
        f"report.{root.root_id}.r{refinement}",
        ObjectIdentity.from_record(config.config_id, config),
        root,
        refinement,
        tuple(
            sorted(
                (
                    ObjectIdentity.from_record(value.result_id, value)
                    for value in by_segment.values()
                ),
                key=lambda value: value.object_id,
            )
        ),
        mode is not None,
        shadow_distance,
        tuple(packets),
        sha256(payload).hexdigest(),
    )
    return report, payload
