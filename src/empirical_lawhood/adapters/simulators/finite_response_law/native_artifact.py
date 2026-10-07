"Closed finite response-law paired-view native artifact; the runtime owns all persistence."

import io
from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

import h5py  # type: ignore[import-untyped]
import numpy as np

from empirical_lawhood.adapters.simulators.prepared_response.native_artifact import ARRAY_NAMES
from empirical_lawhood.adapters.simulators.prepared_response.native_artifact import METADATA as PREPARED_METADATA
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import response_hdf5_read_text, response_hdf5_text, response_hdf5_writer
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256

from .source import FiniteResponseLawAssignedCalibrationCheckpoint, FiniteResponseLawAssignedCalibrationDelivery, FiniteResponseLawAssignedEvaluationCheckpoint, FiniteResponseLawAssignedEvaluationDelivery, FiniteResponseLawCalibrationCheckpoint, FiniteResponseLawCalibrationDelivery, FiniteResponseLawEvaluationCheckpoint, FiniteResponseLawEvaluationDelivery, FiniteResponseLawNativeCheckpoint, FiniteResponseLawNativeDelivery, FiniteResponseLawNativePhaseData

NATIVE_PAIR_SCHEMA = 'empirical-lawhood/simulators/finite-response-law/native-pair-hdf5'
MAXIMUM_PAIR_BYTES = 16 * 1024**2
MAXIMUM_CHECKPOINT_BYTES = 512 * 1024
METADATA = {
    **PREPARED_METADATA,
    "empirical_lawhood_payload_schema": NATIVE_PAIR_SCHEMA,
    "empirical_lawhood_clocks": '{"ticks":"native-step/refinement;one-reference-tick=0.001-native-time","transfer":"trailing-32-reference-tick-operator","handoff":4368,"future-terminal":4560}',
    "numerical_views": '["primary-refinement-1","half-refinement-2"]',
}


@dataclass(frozen=True, slots=True)
class FiniteResponseLawNativePairArtifact(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-native-pair-artifact'
    )
    DELIVERY_TYPE: ClassVar[type[FiniteResponseLawNativeDelivery]] = FiniteResponseLawNativeDelivery
    CHECKPOINT_TYPE: ClassVar[type[FiniteResponseLawNativeCheckpoint]] = FiniteResponseLawNativeCheckpoint
    METADATA: ClassVar[dict[str, str]] = METADATA
    deliveries: tuple[FiniteResponseLawNativeDelivery, ...]
    checkpoints: tuple[ObjectIdentity | None, ...]
    sampled_rows: tuple[int, ...]
    data_sha256: str
    data_bytes: int

    def __post_init__(self) -> None:
        validate_sha256(self.data_sha256, field_name="data_sha256")
        if (
            len(self.deliveries) != 2
            or any(type(d) is not self.DELIVERY_TYPE for d in self.deliveries)
            or len(self.checkpoints) != 2
            or len(self.sampled_rows) != 2
            or tuple(d.refinement for d in self.deliveries) != (1, 2)
            or self.deliveries[0].invocation != self.deliveries[1].invocation
            or self.deliveries[0].frozen_frame_sha256
            != self.deliveries[1].frozen_frame_sha256
            or type(self.data_bytes) is not int
            or not 0 < self.data_bytes <= MAXIMUM_PAIR_BYTES
        ):
            raise ValueError("Finite response-law native artifact changes the paired delivery census")
        for d, c, rows in zip(
            self.deliveries, self.checkpoints, self.sampled_rows, strict=True
        ):
            expected = d.completed_intervals // (16 * d.refinement) + 1
            if (
                type(rows) is not int
                or not 1 <= rows <= 257
                or not (
                    rows == expected
                    or d.disposition == "OBSERVATION_FAILURE"
                    and rows == expected - 1
                )
                or (c is not None) != (d.disposition == "COMPLETE")
                or c is not None
                and (
                    c.object_schema != self.CHECKPOINT_TYPE.SCHEMA
                    or c.object_id != f"{d.occurrence_id}.checkpoint"
                )
            ):
                raise ValueError(
                    "Finite response-law artifact changes its bounded observation/checkpoint inventory"
                )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationPairArtifact(FiniteResponseLawNativePairArtifact):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-calibration-pair-artifact'
    )
    VERSION: ClassVar[str] = '1.0.0'
    DELIVERY_TYPE: ClassVar[type[FiniteResponseLawNativeDelivery]] = FiniteResponseLawCalibrationDelivery
    CHECKPOINT_TYPE: ClassVar[type[FiniteResponseLawNativeCheckpoint]] = FiniteResponseLawCalibrationCheckpoint
    deliveries: tuple[FiniteResponseLawCalibrationDelivery, ...]


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationPairArtifact(FiniteResponseLawNativePairArtifact):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-evaluation-pair-artifact'
    )
    DELIVERY_TYPE: ClassVar[type[FiniteResponseLawNativeDelivery]] = FiniteResponseLawEvaluationDelivery
    CHECKPOINT_TYPE: ClassVar[type[FiniteResponseLawNativeCheckpoint]] = FiniteResponseLawEvaluationCheckpoint
    deliveries: tuple[FiniteResponseLawEvaluationDelivery, ...]


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedCalibrationPairArtifact(FiniteResponseLawNativePairArtifact):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-assigned-calibration-pair-artifact'
    )
    VERSION: ClassVar[str] = '1.0.0'
    DELIVERY_TYPE: ClassVar[type[FiniteResponseLawNativeDelivery]] = (
        FiniteResponseLawAssignedCalibrationDelivery
    )
    CHECKPOINT_TYPE: ClassVar[type[FiniteResponseLawNativeCheckpoint]] = (
        FiniteResponseLawAssignedCalibrationCheckpoint
    )
    deliveries: tuple[FiniteResponseLawAssignedCalibrationDelivery, ...]


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedEvaluationPairArtifact(FiniteResponseLawNativePairArtifact):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-assigned-evaluation-pair-artifact'
    )
    VERSION: ClassVar[str] = '1.0.0'
    DELIVERY_TYPE: ClassVar[type[FiniteResponseLawNativeDelivery]] = FiniteResponseLawAssignedEvaluationDelivery
    CHECKPOINT_TYPE: ClassVar[type[FiniteResponseLawNativeCheckpoint]] = (
        FiniteResponseLawAssignedEvaluationCheckpoint
    )
    deliveries: tuple[FiniteResponseLawAssignedEvaluationDelivery, ...]


def encode_native_pair(
    primary: FiniteResponseLawNativePhaseData,
    half: FiniteResponseLawNativePhaseData,
    *,
    record_type: type[FiniteResponseLawNativePairArtifact] | None = None,
) -> tuple[FiniteResponseLawNativePairArtifact, bytes]:
    values = (primary, half)
    deliveries = tuple(v.delivery for v in values)
    checkpoints = tuple(
        None
        if v.checkpoint is None
        else ObjectIdentity.from_record(v.checkpoint.checkpoint_id, v.checkpoint)
        for v in values
    )
    rows = tuple(len(v.ticks) for v in values)
    if record_type is None:
        record_type = (
            FiniteResponseLawAssignedEvaluationPairArtifact
            if type(primary.delivery) is FiniteResponseLawAssignedEvaluationDelivery
            else FiniteResponseLawAssignedCalibrationPairArtifact
            if type(primary.delivery) is FiniteResponseLawAssignedCalibrationDelivery
            else FiniteResponseLawEvaluationPairArtifact
            if type(primary.delivery) is FiniteResponseLawEvaluationDelivery
            else FiniteResponseLawCalibrationPairArtifact
            if type(primary.delivery) is FiniteResponseLawCalibrationDelivery
            else FiniteResponseLawNativePairArtifact
        )
    record_type(deliveries, checkpoints, rows, "0" * 64, 1)
    stream = io.BytesIO()
    with response_hdf5_writer(stream) as artifact:
        for name, value in sorted(record_type.METADATA.items()):
            response_hdf5_text(artifact, name, value)
        for name, phase in zip(("primary", "half"), values, strict=True):
            group = artifact.create_group(name)
            for key in ARRAY_NAMES:
                group.create_dataset(key, data=getattr(phase, key), track_times=False)
            raw = (
                b"" if phase.checkpoint is None else phase.checkpoint.canonical_bytes()
            )
            if len(raw) > MAXIMUM_CHECKPOINT_BYTES:
                raise ValueError("Finite response-law native checkpoint exceeds artifact bound")
            group.create_dataset(
                "checkpoint", data=np.frombuffer(raw, dtype=np.uint8), track_times=False
            )
    payload = stream.getvalue()
    return record_type(
        deliveries, checkpoints, rows, sha256(payload).hexdigest(), len(payload)
    ), payload


def decode_native_pair(
    report: FiniteResponseLawNativePairArtifact, payload: bytes
) -> tuple[FiniteResponseLawNativePhaseData, FiniteResponseLawNativePhaseData]:
    if (
        type(payload) is not bytes
        or len(payload) != report.data_bytes
        or sha256(payload).hexdigest() != report.data_sha256
    ):
        raise ValueError("Finite response-law native artifact differs from its immutable hash/size")
    phases = []
    with h5py.File(io.BytesIO(payload), "r") as artifact:
        if set(artifact) != {"primary", "half"} or set(artifact.attrs) != set(
            report.METADATA
        ):
            raise ValueError("Finite response-law native artifact changes its closed paired inventory")
        for name, value in report.METADATA.items():
            if response_hdf5_read_text(artifact, name) != value:
                raise ValueError("Finite response-law native artifact changes its units/frames/clocks")
        # Validate the entire topology and every bound before reading either view.
        for i, name in enumerate(("primary", "half")):
            if not isinstance(artifact.get(name, getlink=True), h5py.HardLink):
                raise ValueError(  # noqa: TRY004 - malformed artifact data
                    "Finite response-law native artifact forbids indirect group links"
                )
            group = artifact[name]
            if (
                not isinstance(group, h5py.Group)
                or group.attrs
                or set(group) != {*ARRAY_NAMES, "checkpoint"}
            ):
                raise ValueError("Finite response-law native artifact changes its view inventory")
            rows, count = (
                report.sampled_rows[i],
                report.deliveries[i].completed_intervals,
            )
            expected = {
                "ticks": ((rows,), "<i8"),
                "positions": ((rows, 2, 3, 4, 4), "<c16"),
                "momenta": ((rows, 2, 3, 4, 4), "<c16"),
                "couplings": ((rows, 2), "<f8"),
                "transfer": ((rows, 15, 15), "<f8"),
                "transfer_known": ((rows,), "bool"),
                "realized_trace": ((count, 9), "<f8"),
            }
            for key in group:
                if not isinstance(group.get(key, getlink=True), h5py.HardLink):
                    raise ValueError(  # noqa: TRY004 - malformed artifact data
                        "Finite response-law native artifact forbids indirect dataset links"
                    )
                data = group[key]
                if (
                    not isinstance(data, h5py.Dataset)
                    or data.is_virtual
                    or data.external
                    or data.compression is not None
                    or data.shuffle
                    or data.fletcher32
                    or data.scaleoffset is not None
                    or data.attrs
                ):
                    raise ValueError(
                        "Finite response-law native artifact requires closed unfiltered datasets"
                    )
                if key == "checkpoint":
                    if (
                        data.ndim != 1
                        or data.dtype != np.dtype("uint8")
                        or data.size > MAXIMUM_CHECKPOINT_BYTES
                    ):
                        raise ValueError(
                            "Finite response-law checkpoint buffer exceeds its declared shape/bound"
                        )
                else:
                    shape, dtype = expected[key]
                    if data.shape != shape or data.dtype != np.dtype(dtype):
                        raise ValueError(
                            "Finite response-law native array changes declared shape/dtype"
                        )
        for i, name in enumerate(("primary", "half")):
            group = artifact[name]
            raw = group["checkpoint"][()].tobytes()
            checkpoint = (
                None
                if not raw
                else decode_canonical_bytes(
                    raw, report.CHECKPOINT_TYPE, maximum_bytes=MAXIMUM_CHECKPOINT_BYTES
                )
            )
            identity = (
                None
                if checkpoint is None
                else ObjectIdentity.from_record(checkpoint.checkpoint_id, checkpoint)
            )
            if identity != report.checkpoints[i]:
                raise ValueError(
                    "Finite response-law native checkpoint differs from its artifact identity"
                )
            phases.append(
                FiniteResponseLawNativePhaseData(
                    report.deliveries[i],
                    checkpoint,
                    group["ticks"][()],
                    group["positions"][()],
                    group["momenta"][()],
                    group["couplings"][()],
                    group["transfer"][()],
                    group["transfer_known"][()],
                    group["realized_trace"][()],
                )
            )
    return phases[0], phases[1]
