"""Bounded artifact decoding and complete root/view aggregation by input role."""

from dataclasses import dataclass
from hashlib import sha256
import io
from collections.abc import Mapping

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import CONTEXTS, PreparationRoot, preparation_roots
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import response_hdf5_group, response_hdf5_read_text, response_hdf5_text, response_hdf5_writer
from .contracts import PreparationProjectionRecord
from .projection import _verify_hdf5, read_preparation_projection


Array = npt.NDArray[np.float64]
METHOD_MAXIMUM_BYTES = 16 * 1024**2
METHOD_METADATA = {
    "empirical_lawhood_units": '{"responses,halfwidths,scales":"hilbert-schmidt-native","probabilities,flags":"1","features":"declared-nine-observable-chart"}',
    "empirical_lawhood_frames": '{"response":"primary-preparent-mode;common-parents-views","scope":"exposed-development-only"}',
    "empirical_lawhood_clocks": '{"readout":"64,128,192,256,320-after-handoff","prospective-task":"independent-purpose;concealed-until-prediction-lock"}',
    "empirical_lawhood_keys": '["candidate","root","parent","numerical-view","audit","sign","readout"]',
}
NUMERICAL_SEMANTICS_SCHEMA = 'empirical-lawhood/methods/matrix-preparation/numerical-semantics-arrays-hdf5'
FORECAST_SCHEMA = 'empirical-lawhood/methods/matrix-preparation/locked-development-forecasts-hdf5'
DECISION_SCHEMA = 'empirical-lawhood/methods/matrix-preparation/locked-development-decisions-hdf5'
BENCHMARK_SCHEMA = 'empirical-lawhood/methods/matrix-preparation/description-and-benchmark-assessment-hdf5'
DESCRIPTION_SCHEMA = 'empirical-lawhood/methods/matrix-preparation/observable-description-assessment-hdf5'
TASK_ASSESSMENT_SCHEMA = 'empirical-lawhood/methods/matrix-preparation/development-task-assessment-hdf5'


@dataclass(frozen=True)
class ProjectedContext:
    context: str
    roots: tuple[PreparationRoot, ...]
    reports: tuple[PreparationProjectionRecord, ...]
    role: str
    arrays: Mapping[str, Array]

    @property
    def identities(self) -> tuple[ObjectIdentity, ...]:
        return tuple(ObjectIdentity.from_record(r.report_id, r) for r in self.reports)


def collect_projected_context(
    context: str, values: tuple[tuple[PreparationProjectionRecord, bytes], ...], *, role: str
) -> ProjectedContext:
    if context not in CONTEXTS or role not in ("observable", "privileged", "prospective-task"):
        raise ValueError("projection collection changes context or information role")
    roots = tuple(r for r in preparation_roots() if r.context == context)
    by_slot = {(r.root, r.refinement): (r, payload) for r, payload in values}
    if (
        len(values) != 128
        or len(by_slot) != 128
        or set(by_slot) != {(root, r) for root in roots for r in (1, 2)}
    ):
        raise ValueError("projection collection must retain all 64 roots and both numerical views")
    reports = tuple(by_slot[(root, r)][0] for root in roots for r in (1, 2))
    if len({r.projection_config for r in reports}) != 1:
        raise ValueError("projection collection mixes observer configurations")
    decoded = [
        read_preparation_projection(*by_slot[(root, r)], role=role)
        for root in roots
        for r in (1, 2)
    ]
    arrays = {}
    for key in decoded[0]:
        stacked = np.stack([v[key] for v in decoded]).reshape(64, 2, *decoded[0][key].shape)
        arrays[key] = stacked if key == "preparent_features" else stacked.swapaxes(1, 2)
    return ProjectedContext(context, roots, reports, role, arrays)


def write_method_arrays(
    schema: str, context: str, config: ObjectIdentity, groups: Mapping[str, Mapping[str, Array]]
) -> bytes:
    if context not in CONTEXTS or len(groups) > 32:
        raise ValueError("method arrays exceed their fixed context/group census")
    output = io.BytesIO()
    with response_hdf5_writer(output) as artifact:
        for key, value in {
            "schema": schema,
            "empirical_lawhood_payload_schema": schema,
            "context": context,
            "config_sha256": config.object_fingerprint,
            **METHOD_METADATA,
        }.items():
            response_hdf5_text(artifact, key, value)
        for group_name in sorted(groups):
            group = response_hdf5_group(artifact, group_name)
            for name, array in sorted(groups[group_name].items()):
                group.create_dataset(
                    name, data=np.asarray(array, dtype=np.float64), track_times=False
                )
    payload = output.getvalue()
    if len(payload) > METHOD_MAXIMUM_BYTES:
        raise ValueError("method arrays exceed the declared 16-MiB artifact bound")
    return payload


def read_method_arrays(
    payload: bytes, *, schema: str, context: str, config: ObjectIdentity, expected_sha256: str
) -> dict[str, dict[str, Array]]:
    if len(payload) > METHOD_MAXIMUM_BYTES or sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("method data differs from its immutable report")
    with h5py.File(io.BytesIO(payload), "r") as artifact:
        _verify_hdf5(artifact, METHOD_MAXIMUM_BYTES)
        if (
            response_hdf5_read_text(artifact, "schema") != schema
            or response_hdf5_read_text(artifact, "context") != context
            or response_hdf5_read_text(artifact, "config_sha256") != config.object_fingerprint
        ):
            raise ValueError("method data changes its context/config/information role")
        result = {}
        for name in artifact:
            group = artifact[name]
            if not isinstance(group, h5py.Group):
                raise ValueError("method data requires the exact bounded group schema")
            arrays = {}
            for key in group:
                value = group[key]
                if not isinstance(value, h5py.Dataset) or value.dtype != np.dtype("float64"):
                    raise ValueError("method arrays must be nonexecuting float64 data")
                arrays[key] = np.asarray(value[...])
            result[name] = arrays
    return result
