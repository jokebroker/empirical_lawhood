"""Paired-view native custody codec; no execution or selection is performed here.

Both reports remain independent deliveries of one root/phase. A failed view
is retained with its actual arrays and absent checkpoint. This container does
not manufacture a delivery for an unentered phase or grant continuation.
"""

from dataclasses import dataclass
from hashlib import sha256
import io
from typing import ClassVar

import h5py  # type: ignore[import-untyped]
import numpy as np

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import response_hdf5_read_text, response_hdf5_text, response_hdf5_writer

from .native_artifact import MAXIMUM_PHASE_BYTES, METADATA, PreparedNativePhaseArtifact, decode_prepared_native_phase, encode_prepared_native_phase
from .source import PreparedNativePhaseData
from .native_tasks import PreparedNativeInvocation


NATIVE_PAIR_SCHEMA = 'empirical-lawhood/simulators/prepared-response/native-pair-hdf5'
MAXIMUM_PAIR_BYTES = 2 * MAXIMUM_PHASE_BYTES + 64 * 1024
PAIR_METADATA = {
    **METADATA,
    "empirical_lawhood_payload_schema": NATIVE_PAIR_SCHEMA,
    "numerical_views": '["primary-refinement-1","half-refinement-2"]',
    "container_layout": "two-unfiltered-uint8-native-phase-hdf5-buffers",
}


@dataclass(frozen=True, slots=True)
class PreparedNativePairArtifact(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-response/prepared-native-pair-artifact'
    purpose: str
    views: tuple[PreparedNativePhaseArtifact, ...]
    data_sha256: str
    data_bytes: int

    def __post_init__(self) -> None:
        validate_sha256(self.data_sha256, field_name="data_sha256")
        if (
            type(self.views) is not tuple
            or len(self.views) != 2
            or tuple(v.delivery.refinement for v in self.views) != (1, 2)
            or type(self.data_bytes) is not int
            or not sum(v.data_bytes for v in self.views) < self.data_bytes <= MAXIMUM_PAIR_BYTES
        ):
            raise ValueError("native pair changes its exact ordered two-view byte census")
        first, second = (v.delivery for v in self.views)
        if (
            first.source_spec,
            first.root,
            first.phase,
            first.parent,
            first.word,
            first.common_start,
            first.start_tick,
            first.requested_end_tick,
        ) != (
            second.source_spec,
            second.root,
            second.phase,
            second.parent,
            second.word,
            second.common_start,
            second.start_tick,
            second.requested_end_tick,
        ):
            raise ValueError("native pair mixes source/root/phase/action/common-start identities")
        if (
            tuple(v.delivery.occurrence_id for v in self.views)
            != self.invocation.native_occurrence_ids
        ):
            raise ValueError("native pair changes an exact view occurrence/purpose identity")

    @property
    def invocation(self) -> PreparedNativeInvocation:
        delivery = self.views[0].delivery
        return PreparedNativeInvocation(
            delivery.source_spec,
            delivery.root,
            delivery.phase,
            delivery.parent,
            delivery.word,
            self.purpose,
        )

    @property
    def result_id(self) -> str:
        return f"{self.views[0].delivery.occurrence_id[:-3]}.paired-native-artifact"


def encode_prepared_native_pair(
    primary: PreparedNativePhaseData,
    half: PreparedNativePhaseData,
    *,
    purpose: str,
) -> tuple[PreparedNativePairArtifact, bytes]:
    reports_and_bytes = tuple(encode_prepared_native_phase(v) for v in (primary, half))
    # Validate the identity join before building the outer container.
    reports = tuple(v[0] for v in reports_and_bytes)
    PreparedNativePairArtifact(purpose, reports, "0" * 64, sum(r.data_bytes for r in reports) + 1)
    stream = io.BytesIO()
    with response_hdf5_writer(stream) as artifact:
        for name, value in sorted(PAIR_METADATA.items()):
            response_hdf5_text(artifact, name, value)
        for name, (_, payload) in zip(("primary", "half"), reports_and_bytes, strict=True):
            artifact.create_dataset(
                name, data=np.frombuffer(payload, dtype=np.uint8), track_times=False
            )
    payload = stream.getvalue()
    return PreparedNativePairArtifact(
        purpose, reports, sha256(payload).hexdigest(), len(payload)
    ), payload


def decode_prepared_native_pair(
    report: PreparedNativePairArtifact, payload: bytes
) -> tuple[PreparedNativePhaseData, PreparedNativePhaseData]:
    if (
        type(payload) is not bytes
        or len(payload) != report.data_bytes
        or len(payload) > MAXIMUM_PAIR_BYTES
        or sha256(payload).hexdigest() != report.data_sha256
    ):
        raise ValueError("native pair bytes differ from their immutable artifact record")
    with h5py.File(io.BytesIO(payload), "r") as artifact:
        if set(artifact) != {"primary", "half"} or set(artifact.attrs) != set(PAIR_METADATA):
            raise ValueError("native pair changes its closed dataset/metadata census")
        for name, expected in PAIR_METADATA.items():
            if response_hdf5_read_text(artifact, name) != expected:
                raise ValueError("native pair changes its units/frames/clocks/view metadata")
        for name, view in zip(("primary", "half"), report.views, strict=True):
            if not isinstance(artifact.get(name, getlink=True), h5py.HardLink):
                raise ValueError("native pair forbids external and symbolic HDF5 links")
            dataset = artifact[name]
            if (
                not isinstance(dataset, h5py.Dataset)
                or dataset.shape != (view.data_bytes,)
                or dataset.dtype != np.dtype("uint8")
                or dataset.is_virtual
                or dataset.external
                or dataset.compression is not None
                or dataset.shuffle
                or dataset.fletcher32
                or dataset.scaleoffset is not None
                or dataset.attrs
            ):
                raise ValueError("native pair changes its bounded unfiltered byte-buffer layout")
        # Validate the complete outer topology before reading either view.
        primary = artifact["primary"][()].tobytes()
        half = artifact["half"][()].tobytes()
    return (
        decode_prepared_native_phase(report.views[0], primary),
        decode_prepared_native_phase(report.views[1], half),
    )
