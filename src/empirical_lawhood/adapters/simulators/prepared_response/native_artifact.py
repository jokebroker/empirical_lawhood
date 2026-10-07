"""Bounded, deterministic native-phase artifacts for the production source port.

This codec performs no source contact or filesystem I/O. The runtime artifact
port owns publication, custody and external storage. A decoded checkpoint is
the same phase identity; decoding does not reacquire or retry scientific work.
"""

from dataclasses import dataclass
from hashlib import sha256
import io
from typing import ClassVar

import h5py  # type: ignore[import-untyped]
import numpy as np

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import response_hdf5_read_text, response_hdf5_text, response_hdf5_writer

from .source import PreparedNativeCheckpoint, PreparedNativeDelivery, PreparedNativePhaseData

NATIVE_PHASE_SCHEMA = 'empirical-lawhood/simulators/prepared-response/native-phase-hdf5'
MAXIMUM_PHASE_BYTES = 8 * 1024 * 1024
MAXIMUM_CHECKPOINT_BYTES = 512 * 1024
MAXIMUM_DELIVERY_BYTES = 64 * 1024
ARRAY_NAMES = (
    "couplings",
    "momenta",
    "positions",
    "realized_trace",
    "ticks",
    "transfer",
    "transfer_known",
)
METADATA = {
    "empirical_lawhood_payload_schema": NATIVE_PHASE_SCHEMA,
    "empirical_lawhood_units": '{"positions":"native-hs-position","momenta":"native-hs-momentum;kinetic-mass=1","couplings":"native-reduced","ticks":"reference-tick","transfer":"dimensionless","realized_trace":"declared-native-clock-coupling-force-columns"}',
    "empirical_lawhood_frames": '{"positions,momenta":"sector,spatial,matrix-row,matrix-column","transfer":"traceless-Hermitian-receiver-basis","force":"common-start-frozen-primary-two-port"}',
    "empirical_lawhood_clocks": '{"ticks":"native-step/refinement;one-reference-tick=0.001-native-time","transfer":"trailing-32-reference-tick-operator","handoff":"landmark+272"}',
    "empirical_lawhood_keys": '["source-spec","root","phase","parent","purpose","word","refinement"]',
    "realized_trace_columns": '["native-step-before","alpha-x-before","alpha-y-before","alpha-x-after","alpha-y-after","first-kick-force-m1","first-kick-force-m2","second-kick-force-m1","second-kick-force-m2"]',
}


@dataclass(frozen=True, slots=True)
class PreparedNativePhaseArtifact(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-response/prepared-native-phase-artifact'
    delivery: PreparedNativeDelivery
    data_sha256: str
    data_bytes: int
    sampled_rows: int
    checkpoint: ObjectIdentity | None

    def __post_init__(self) -> None:
        validate_sha256(self.data_sha256, field_name="data_sha256")
        expected = self.delivery.completed_intervals // (16 * self.delivery.refinement) + 1
        if (
            type(self.data_bytes) is not int
            or not 0 < self.data_bytes <= MAXIMUM_PHASE_BYTES
            or type(self.sampled_rows) is not int
            or not 1 <= self.sampled_rows <= 257
            or not (
                self.sampled_rows == expected
                or self.delivery.disposition == "OBSERVATION_FAILURE"
                and self.sampled_rows == expected - 1
            )
            or (self.checkpoint is not None) != (self.delivery.disposition == "COMPLETE")
            or self.checkpoint is not None
            and self.checkpoint.object_schema != PreparedNativeCheckpoint.SCHEMA
        ):
            raise ValueError("native phase artifact changes its bounded delivery/checkpoint roster")

    @property
    def result_id(self) -> str:
        return f"{self.delivery.occurrence_id}.native-artifact"


def encode_prepared_native_phase(
    value: PreparedNativePhaseData,
) -> tuple[PreparedNativePhaseArtifact, bytes]:
    stream = io.BytesIO()
    with response_hdf5_writer(stream) as artifact:
        for name, text in sorted(METADATA.items()):
            response_hdf5_text(artifact, name, text)
        for name in ARRAY_NAMES:
            artifact.create_dataset(name, data=getattr(value, name), track_times=False)
        delivery = value.delivery.canonical_bytes()
        checkpoint = b"" if value.checkpoint is None else value.checkpoint.canonical_bytes()
        if len(delivery) > MAXIMUM_DELIVERY_BYTES or len(checkpoint) > MAXIMUM_CHECKPOINT_BYTES:
            raise ValueError("native control record exceeds its declared artifact bound")
        for name, payload in (("delivery_record", delivery), ("checkpoint_record", checkpoint)):
            artifact.create_dataset(
                name, data=np.frombuffer(payload, dtype=np.uint8), track_times=False
            )
    payload = stream.getvalue()
    return PreparedNativePhaseArtifact(
        value.delivery,
        sha256(payload).hexdigest(),
        len(payload),
        len(value.ticks),
        None
        if value.checkpoint is None
        else ObjectIdentity.from_record(value.checkpoint.checkpoint_id, value.checkpoint),
    ), payload


def decode_prepared_native_phase(
    report: PreparedNativePhaseArtifact, payload: bytes
) -> PreparedNativePhaseData:
    if (
        type(payload) is not bytes
        or len(payload) != report.data_bytes
        or len(payload) > MAXIMUM_PHASE_BYTES
        or sha256(payload).hexdigest() != report.data_sha256
    ):
        raise ValueError("native phase bytes differ from their immutable artifact record")
    rows, count = report.sampled_rows, report.delivery.completed_intervals
    expected = {
        "ticks": ((rows,), "<i8"),
        "positions": ((rows, 2, 3, 4, 4), "<c16"),
        "momenta": ((rows, 2, 3, 4, 4), "<c16"),
        "couplings": ((rows, 2), "<f8"),
        "transfer": ((rows, 15, 15), "<f8"),
        "transfer_known": ((rows,), "bool"),
        "realized_trace": ((count, 9), "<f8"),
    }
    with h5py.File(io.BytesIO(payload), "r") as artifact:
        if set(artifact) != {*expected, "delivery_record", "checkpoint_record"} or set(
            artifact.attrs
        ) != set(METADATA):
            raise ValueError("native artifact has an extra or missing dataset/identity attribute")
        for name, text in METADATA.items():
            if response_hdf5_read_text(artifact, name) != text:
                raise ValueError("native artifact changes its declared units/frames/clocks")
        for name in artifact:
            if not isinstance(artifact.get(name, getlink=True), h5py.HardLink):
                raise ValueError("native artifact cannot contain an external or symbolic HDF5 link")
            dataset = artifact[name]
            if (
                not isinstance(dataset, h5py.Dataset)
                or dataset.is_virtual
                or dataset.external
                or dataset.compression is not None
                or dataset.shuffle
                or dataset.fletcher32
                or dataset.scaleoffset is not None
                or dataset.attrs
            ):
                raise ValueError("native artifact changes its closed unfiltered dataset layout")
            if name in expected:
                shape, dtype = expected[name]
                if dataset.shape != shape or dataset.dtype != np.dtype(dtype):
                    raise ValueError("native dataset has an undeclared shape or dtype")
            elif (
                dataset.dtype != np.dtype("uint8")
                or dataset.ndim != 1
                or dataset.size
                > (
                    MAXIMUM_DELIVERY_BYTES
                    if name == "delivery_record"
                    else MAXIMUM_CHECKPOINT_BYTES
                )
            ):
                raise ValueError(
                    "native canonical record has an unbounded or unexpected byte shape"
                )
        # Decode no array or variable-sized object before the entire bounded
        # topology above has passed, including every control-record buffer.
        delivery = decode_canonical_bytes(
            artifact["delivery_record"][()].tobytes(),
            PreparedNativeDelivery,
            maximum_bytes=MAXIMUM_DELIVERY_BYTES,
        )
        if delivery != report.delivery:
            raise ValueError("native HDF5 delivery differs from its immutable result")
        checkpoint_payload = artifact["checkpoint_record"][()].tobytes()
        checkpoint = (
            None
            if not checkpoint_payload
            else decode_canonical_bytes(
                checkpoint_payload,
                PreparedNativeCheckpoint,
                maximum_bytes=MAXIMUM_CHECKPOINT_BYTES,
            )
        )
        identity = (
            None
            if checkpoint is None
            else ObjectIdentity.from_record(checkpoint.checkpoint_id, checkpoint)
        )
        if identity != report.checkpoint:
            raise ValueError("native HDF5 checkpoint differs from its immutable result")
        arrays = {name: np.asarray(artifact[name][()]) for name in expected}
    return PreparedNativePhaseData(
        delivery,
        checkpoint,
        arrays["ticks"],
        arrays["positions"],
        arrays["momenta"],
        arrays["couplings"],
        arrays["transfer"],
        arrays["transfer_known"],
        arrays["realized_trace"],
    )
