"""Safe table/model codecs and the audited H5AD byte envelope."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
from typing import BinaryIO, cast

import h5py  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
import pyarrow as pa  # type: ignore[import-untyped]
from safetensors.numpy import load as load_safetensors
from safetensors.numpy import save as save_safetensors

from .analysis import (
    LinearResponseModel,
    ModelFamily,
    PrefixAdmissionModel,
    Standardization,
    TargetFeatureMatrix,
)
from .contracts import VirtualCellContractError
from .dataset import ControlReservoirArrays, ResponseSummaryArrays
from .records import (
    CONTROL_RESERVOIR_TABLE_SCHEMA,
    MODEL_SAFETENSORS_SCHEMA,
    OFFICIAL_METRICS_TABLE_SCHEMA,
    PREDICTED_MEAN_TABLE_SCHEMA,
    PREDICTION_H5AD_ENVELOPE_SCHEMA,
    RESPONSE_SUMMARY_TABLE_SCHEMA,
    TARGET_FEATURE_TABLE_SCHEMA,
)


_VECTOR_DTYPE = np.dtype("<f8")
_CONTROL_VECTOR_DTYPE = np.dtype("<f4")
_ENVELOPE_CHUNK_BYTES = 8 * 1024**2


def _schema_metadata(
    *,
    payload_schema: str,
    logical_types: dict[str, str],
    units: dict[str, str],
    frames: dict[str, str],
    clocks: dict[str, str],
    keys: tuple[str, ...],
) -> dict[bytes, bytes]:
    def encoded(value: object) -> bytes:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    return {
        b"empirical_lawhood_payload_schema": payload_schema.encode("utf-8"),
        b"empirical_lawhood_logical_types": encoded(logical_types),
        b"empirical_lawhood_units": encoded(units),
        b"empirical_lawhood_frames": encoded(frames),
        b"empirical_lawhood_clocks": encoded(clocks),
        b"empirical_lawhood_keys": encoded(list(keys)),
    }


def response_summary_arrow_schema() -> pa.Schema:
    names = ("target_id", "batch_id", "cell_count", "mean_le_f64", "variance_le_f64")
    return pa.schema(
        [
            pa.field("target_id", pa.string(), nullable=False),
            pa.field("batch_id", pa.string(), nullable=False),
            pa.field("cell_count", pa.int64(), nullable=False),
            pa.field("mean_le_f64", pa.binary(), nullable=False),
            pa.field("variance_le_f64", pa.binary(), nullable=False),
        ],
        metadata=_schema_metadata(
            payload_schema=RESPONSE_SUMMARY_TABLE_SCHEMA,
            logical_types={
                "target_id": "nominal-target-gene",
                "batch_id": "observed-batch-ceiling",
                "cell_count": "nested-cell-count",
                "mean_le_f64": "dense-gene-mean-vector-little-endian-float64",
                "variance_le_f64": "dense-gene-conditional-variance-vector-little-endian-float64",
            },
            units={name: "1" for name in names},
            frames={
                "target_id": "source-target-registry",
                "batch_id": "source-batch-registry",
                "cell_count": "nested-cell-view",
                "mean_le_f64": "source-gene-order",
                "variance_le_f64": "source-gene-order",
            },
            clocks={},
            keys=("target_id", "batch_id"),
        ),
    )


def target_feature_arrow_schema() -> pa.Schema:
    names = ("target_id", "feature_vector_le_f64")
    return pa.schema(
        [
            pa.field("target_id", pa.string(), nullable=False),
            pa.field("feature_vector_le_f64", pa.binary(), nullable=False),
        ],
        metadata=_schema_metadata(
            payload_schema=TARGET_FEATURE_TABLE_SCHEMA,
            logical_types={
                "target_id": "nominal-target-gene",
                "feature_vector_le_f64": "outcome-blind-feature-vector-little-endian-float64",
            },
            units={name: "1" for name in names},
            frames={
                "target_id": "source-target-registry",
                "feature_vector_le_f64": "frozen-feature-registry",
            },
            clocks={},
            keys=("target_id",),
        ),
    )


def control_reservoir_arrow_schema() -> pa.Schema:
    names = ("row_id", "batch_id", "values_le_f32")
    return pa.schema(
        [
            pa.field("row_id", pa.string(), nullable=False),
            pa.field("batch_id", pa.string(), nullable=False),
            pa.field("values_le_f32", pa.binary(), nullable=False),
        ],
        metadata=_schema_metadata(
            payload_schema=CONTROL_RESERVOIR_TABLE_SCHEMA,
            logical_types={
                "row_id": "nested-source-cell-id",
                "batch_id": "observed-batch-ceiling",
                "values_le_f32": "dense-control-expression-vector-little-endian-float32",
            },
            units={name: "1" for name in names},
            frames={
                "row_id": "source-row-order",
                "batch_id": "source-batch-registry",
                "values_le_f32": "source-gene-order",
            },
            clocks={},
            keys=("row_id",),
        ),
    )


def predicted_mean_arrow_schema() -> pa.Schema:
    names = ("target_id", "predicted_mean_le_f64", "uncertainty_le_f64")
    return pa.schema(
        [
            pa.field("target_id", pa.string(), nullable=False),
            pa.field("predicted_mean_le_f64", pa.binary(), nullable=False),
            pa.field("uncertainty_le_f64", pa.binary(), nullable=False),
        ],
        metadata=_schema_metadata(
            payload_schema=PREDICTED_MEAN_TABLE_SCHEMA,
            logical_types={
                "target_id": "nominal-target-gene",
                "predicted_mean_le_f64": "predicted-log1p-fixed-total-mean",
                "uncertainty_le_f64": "training-residual-rms-gene-vector",
            },
            units={name: "1" for name in names},
            frames={
                "target_id": "source-target-registry",
                "predicted_mean_le_f64": "source-gene-order",
                "uncertainty_le_f64": "source-gene-order",
            },
            clocks={},
            keys=("target_id",),
        ),
    )


def official_metrics_arrow_schema() -> pa.Schema:
    names = ("target_id", "des", "pds", "mae")
    return pa.schema(
        [
            pa.field("target_id", pa.string(), nullable=False),
            pa.field("des", pa.float64(), nullable=False),
            pa.field("pds", pa.float64(), nullable=False),
            pa.field("mae", pa.float64(), nullable=False),
        ],
        metadata=_schema_metadata(
            payload_schema=OFFICIAL_METRICS_TABLE_SCHEMA,
            logical_types={
                "target_id": "nominal-target-gene",
                "des": "cell-eval-discrimination-score-l1",
                "pds": "cell-eval-overlap-at-N",
                "mae": "cell-eval-pseudobulk-mean-absolute-error",
            },
            units={name: "1" for name in names},
            frames={
                "target_id": "evaluation-target-registry",
                "des": "official-vcc-2025-metric",
                "pds": "official-vcc-2025-metric",
                "mae": "official-vcc-2025-metric",
            },
            clocks={},
            keys=("target_id",),
        ),
    )


def _write_table(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return cast(bytes, sink.getvalue().to_pybytes())


def _read_table(payload: bytes, *, expected_schema: pa.Schema) -> pa.Table:
    try:
        reader = pa.ipc.open_file(pa.BufferReader(payload))
        table = reader.read_all()
    except (pa.ArrowException, OSError, ValueError) as error:
        raise VirtualCellContractError("Arrow payload is invalid") from error
    if table.schema != expected_schema:
        raise VirtualCellContractError("Arrow payload differs from its exact schema")
    return table


def _vector_bytes(value: NDArray[np.float64]) -> bytes:
    return np.ascontiguousarray(value, dtype=_VECTOR_DTYPE).tobytes(order="C")


def _decode_vector(value: bytes, *, length: int) -> NDArray[np.float64]:
    expected = length * _VECTOR_DTYPE.itemsize
    if len(value) != expected:
        raise VirtualCellContractError("encoded vector has the wrong byte length")
    return np.frombuffer(value, dtype=_VECTOR_DTYPE).copy()


def _control_vector_bytes(value: NDArray[np.float32]) -> bytes:
    return np.ascontiguousarray(value, dtype=_CONTROL_VECTOR_DTYPE).tobytes(order="C")


def _decode_control_vector(value: bytes, *, length: int) -> NDArray[np.float32]:
    expected = length * _CONTROL_VECTOR_DTYPE.itemsize
    if len(value) != expected:
        raise VirtualCellContractError("encoded control vector has the wrong byte length")
    return np.frombuffer(value, dtype=_CONTROL_VECTOR_DTYPE).copy()


def encode_response_summary(summary: ResponseSummaryArrays) -> bytes:
    table = pa.Table.from_arrays(
        [
            pa.array(summary.target_ids, type=pa.string()),
            pa.array(summary.batch_ids, type=pa.string()),
            pa.array(summary.cell_counts, type=pa.int64()),
            pa.array((_vector_bytes(value) for value in summary.means), type=pa.binary()),
            pa.array((_vector_bytes(value) for value in summary.variances), type=pa.binary()),
        ],
        schema=response_summary_arrow_schema(),
    )
    return _write_table(table)


def decode_response_summary(
    payload: bytes,
    *,
    gene_ids: tuple[str, ...],
    normalization: str,
    normalization_target_sum: float | None = None,
) -> ResponseSummaryArrays:
    table = _read_table(payload, expected_schema=response_summary_arrow_schema())
    target_ids = tuple(cast(str, value.as_py()) for value in table["target_id"])
    batch_ids = tuple(cast(str, value.as_py()) for value in table["batch_id"])
    if len(set(zip(target_ids, batch_ids, strict=True))) != len(target_ids):
        raise VirtualCellContractError("response summary contains duplicate target/batch keys")
    group_ids = tuple(
        f"target={target}|batch={batch}"
        for target, batch in zip(target_ids, batch_ids, strict=True)
    )
    order = tuple(sorted(range(len(group_ids)), key=group_ids.__getitem__))
    if order != tuple(range(len(group_ids))):
        raise VirtualCellContractError("response summary target/batch keys are not canonical")
    means = np.stack(
        [
            _decode_vector(cast(bytes, value.as_py()), length=len(gene_ids))
            for value in table["mean_le_f64"]
        ]
    )
    variances = np.stack(
        [
            _decode_vector(cast(bytes, value.as_py()), length=len(gene_ids))
            for value in table["variance_le_f64"]
        ]
    )
    return ResponseSummaryArrays(
        group_ids=group_ids,
        target_ids=target_ids,
        batch_ids=batch_ids,
        gene_ids=gene_ids,
        normalization=normalization,
        normalization_target_sum=normalization_target_sum,
        cell_counts=np.asarray(table["cell_count"].to_numpy(), dtype=np.int64),
        means=means,
        variances=variances,
    )


def encode_control_reservoir(reservoir: ControlReservoirArrays) -> bytes:
    return _write_table(
        pa.Table.from_arrays(
            [
                pa.array(reservoir.row_ids, type=pa.string()),
                pa.array(reservoir.batch_ids, type=pa.string()),
                pa.array(
                    (_control_vector_bytes(value) for value in reservoir.values),
                    type=pa.binary(),
                ),
            ],
            schema=control_reservoir_arrow_schema(),
        )
    )


def decode_control_reservoir(
    payload: bytes,
    *,
    gene_ids: tuple[str, ...],
    normalization_target_sum: float,
) -> ControlReservoirArrays:
    table = _read_table(payload, expected_schema=control_reservoir_arrow_schema())
    row_ids = tuple(cast(str, value.as_py()) for value in table["row_id"])
    batch_ids = tuple(cast(str, value.as_py()) for value in table["batch_id"])
    if tuple(sorted(row_ids)) != row_ids or len(set(row_ids)) != len(row_ids):
        raise VirtualCellContractError("control reservoir row IDs are not canonical")
    values = np.stack(
        [
            _decode_control_vector(cast(bytes, value.as_py()), length=len(gene_ids))
            for value in table["values_le_f32"]
        ]
    )
    return ControlReservoirArrays(
        row_ids=row_ids,
        batch_ids=batch_ids,
        gene_ids=gene_ids,
        normalization="log1p-fixed-total",
        normalization_target_sum=normalization_target_sum,
        values=values,
    )


def encode_target_features(features: TargetFeatureMatrix) -> bytes:
    return _write_table(
        pa.Table.from_arrays(
            [
                pa.array(features.target_ids, type=pa.string()),
                pa.array((_vector_bytes(value) for value in features.values), type=pa.binary()),
            ],
            schema=target_feature_arrow_schema(),
        )
    )


def decode_target_features(
    payload: bytes,
    *,
    feature_ids: tuple[str, ...],
    provenance_sha256: str,
) -> TargetFeatureMatrix:
    table = _read_table(payload, expected_schema=target_feature_arrow_schema())
    target_ids = tuple(cast(str, value.as_py()) for value in table["target_id"])
    if len(set(target_ids)) != len(target_ids):
        raise VirtualCellContractError("target feature table contains duplicate targets")
    matrix = np.stack(
        [
            _decode_vector(cast(bytes, value.as_py()), length=len(feature_ids))
            for value in table["feature_vector_le_f64"]
        ]
    )
    return TargetFeatureMatrix(
        target_ids=target_ids,
        feature_ids=feature_ids,
        values=matrix,
        provenance_sha256=provenance_sha256,
    )


def encode_predicted_means(
    *,
    target_ids: tuple[str, ...],
    means: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
) -> bytes:
    if means.ndim != 2 or means.shape[0] != len(target_ids):
        raise ValueError("predicted mean array is not target-aligned")
    if uncertainty.shape == (means.shape[1],):
        uncertainty = np.broadcast_to(uncertainty, means.shape)
    if uncertainty.shape != means.shape:
        raise ValueError("prediction uncertainty is not target/gene aligned")
    return _write_table(
        pa.Table.from_arrays(
            [
                pa.array(target_ids, type=pa.string()),
                pa.array((_vector_bytes(value) for value in means), type=pa.binary()),
                pa.array((_vector_bytes(value) for value in uncertainty), type=pa.binary()),
            ],
            schema=predicted_mean_arrow_schema(),
        )
    )


def decode_predicted_means(
    payload: bytes,
    *,
    gene_count: int,
) -> tuple[tuple[str, ...], NDArray[np.float64], NDArray[np.float64]]:
    table = _read_table(payload, expected_schema=predicted_mean_arrow_schema())
    target_ids = tuple(cast(str, value.as_py()) for value in table["target_id"])
    if len(set(target_ids)) != len(target_ids):
        raise VirtualCellContractError("predicted mean table contains duplicate targets")
    means = np.stack(
        [
            _decode_vector(cast(bytes, value.as_py()), length=gene_count)
            for value in table["predicted_mean_le_f64"]
        ]
    )
    uncertainty = np.stack(
        [
            _decode_vector(cast(bytes, value.as_py()), length=gene_count)
            for value in table["uncertainty_le_f64"]
        ]
    )
    return target_ids, means, uncertainty


def encode_official_metrics(
    *,
    target_ids: tuple[str, ...],
    des: NDArray[np.float64],
    pds: NDArray[np.float64],
    mae: NDArray[np.float64],
) -> bytes:
    if tuple(sorted(set(target_ids))) != target_ids:
        raise ValueError("official metric target IDs must be sorted and unique")
    expected = (len(target_ids),)
    if any(values.shape != expected for values in (des, pds, mae)):
        raise ValueError("official metrics are not target aligned")
    if any(not np.all(np.isfinite(values)) for values in (des, pds, mae)):
        raise ValueError("official metrics contain nonfinite values")
    return _write_table(
        pa.Table.from_arrays(
            [
                pa.array(target_ids, type=pa.string()),
                pa.array(des, type=pa.float64()),
                pa.array(pds, type=pa.float64()),
                pa.array(mae, type=pa.float64()),
            ],
            schema=official_metrics_arrow_schema(),
        )
    )


def decode_official_metrics(
    payload: bytes,
) -> tuple[
    tuple[str, ...],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    table = _read_table(payload, expected_schema=official_metrics_arrow_schema())
    target_ids = tuple(cast(str, value.as_py()) for value in table["target_id"])
    if tuple(sorted(set(target_ids))) != target_ids:
        raise VirtualCellContractError("official metric target IDs are not canonical")
    arrays = tuple(
        np.asarray(table[name].to_numpy(), dtype=np.float64) for name in ("des", "pds", "mae")
    )
    return target_ids, arrays[0], arrays[1], arrays[2]


def encode_linear_model(model: LinearResponseModel) -> bytes:
    standardization = model.standardization
    prefix = model.prefix_admission_model
    basis = (
        np.empty((0, len(model.gene_ids)), dtype=np.float64)
        if model.response_basis is None
        else model.response_basis
    )
    tensors = {
        "coefficients": np.asarray(model.coefficients, dtype=np.float64),
        "intercept": np.asarray(model.intercept, dtype=np.float64),
        "residual_scale": np.asarray(model.residual_scale, dtype=np.float64),
        "response_basis": np.asarray(basis, dtype=np.float64),
        "standardization_mean": (
            np.empty((0,), dtype=np.float64)
            if standardization is None
            else np.asarray(standardization.mean, dtype=np.float64)
        ),
        "standardization_scale": (
            np.empty((0,), dtype=np.float64)
            if standardization is None
            else np.asarray(standardization.scale, dtype=np.float64)
        ),
        "prefix_coefficients": (
            np.empty((0,), dtype=np.float64)
            if prefix is None
            else np.asarray(prefix.coefficients, dtype=np.float64)
        ),
        "prefix_standardization_mean": (
            np.empty((0,), dtype=np.float64)
            if prefix is None
            else np.asarray(prefix.standardization.mean, dtype=np.float64)
        ),
        "prefix_standardization_scale": (
            np.empty((0,), dtype=np.float64)
            if prefix is None
            else np.asarray(prefix.standardization.scale, dtype=np.float64)
        ),
    }
    metadata = {
        "empirical_lawhood_payload_schema": MODEL_SAFETENSORS_SCHEMA,
        "model_id": model.model_id,
        "family": model.family.value,
        "feature_ids": json.dumps(model.feature_ids, separators=(",", ":")),
        "gene_ids": json.dumps(model.gene_ids, separators=(",", ":")),
        "admission_required": "true" if model.admission_required else "false",
        "admission_feature_count": str(model.admission_feature_count),
        "prefix_model_id": "" if prefix is None else prefix.model_id,
        "prefix_feature_ids": json.dumps(
            () if prefix is None else prefix.feature_ids,
            separators=(",", ":"),
        ),
        "prefix_intercept": "" if prefix is None else float(prefix.intercept).hex(),
        "prefix_alpha": "" if prefix is None else float(prefix.alpha).hex(),
    }
    return save_safetensors(tensors, metadata=metadata)


def decode_linear_model(payload: bytes) -> LinearResponseModel:
    try:
        tensors = load_safetensors(payload)
        header_size = int.from_bytes(payload[:8], "little")
        header = json.loads(payload[8 : 8 + header_size])
        metadata = header["__metadata__"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise VirtualCellContractError("model SafeTensors payload is invalid") from error
    if metadata.get("empirical_lawhood_payload_schema") != MODEL_SAFETENSORS_SCHEMA:
        raise VirtualCellContractError("model SafeTensors schema metadata differs")
    expected_tensors = {
        "coefficients",
        "intercept",
        "residual_scale",
        "response_basis",
        "standardization_mean",
        "standardization_scale",
        "prefix_coefficients",
        "prefix_standardization_mean",
        "prefix_standardization_scale",
    }
    if set(tensors) != expected_tensors:
        raise VirtualCellContractError("model SafeTensors tensor roster differs")
    try:
        feature_ids = tuple(json.loads(metadata["feature_ids"]))
        gene_ids = tuple(json.loads(metadata["gene_ids"]))
        family = ModelFamily(metadata["family"])
        admission_required = metadata["admission_required"] == "true"
        admission_feature_count = int(metadata["admission_feature_count"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise VirtualCellContractError("model SafeTensors metadata is invalid") from error
    mean = np.asarray(tensors["standardization_mean"], dtype=np.float64)
    scale = np.asarray(tensors["standardization_scale"], dtype=np.float64)
    if (mean.size == 0) != (scale.size == 0):
        raise VirtualCellContractError("model standardization tensors are incomplete")
    standardization = None if mean.size == 0 else Standardization(mean=mean, scale=scale)
    prefix_coefficients = np.asarray(tensors["prefix_coefficients"], dtype=np.float64)
    prefix_mean = np.asarray(tensors["prefix_standardization_mean"], dtype=np.float64)
    prefix_scale = np.asarray(tensors["prefix_standardization_scale"], dtype=np.float64)
    if not (
        (prefix_coefficients.size == 0 and prefix_mean.size == 0 and prefix_scale.size == 0)
        or prefix_coefficients.shape == prefix_mean.shape == prefix_scale.shape
    ):
        raise VirtualCellContractError("prefix-admission tensors are incomplete")
    prefix_model: PrefixAdmissionModel | None
    if prefix_coefficients.size == 0:
        prefix_model = None
    else:
        try:
            prefix_feature_ids = tuple(json.loads(metadata["prefix_feature_ids"]))
            prefix_model = PrefixAdmissionModel(
                model_id=metadata["prefix_model_id"],
                feature_ids=cast(tuple[str, ...], prefix_feature_ids),
                standardization=Standardization(mean=prefix_mean, scale=prefix_scale),
                coefficients=prefix_coefficients,
                intercept=float.fromhex(metadata["prefix_intercept"]),
                alpha=float.fromhex(metadata["prefix_alpha"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise VirtualCellContractError("prefix-admission metadata is invalid") from error
    basis_array = np.asarray(tensors["response_basis"], dtype=np.float64)
    basis = None if basis_array.shape[0] == 0 else basis_array
    model_id = metadata.get("model_id")
    if not isinstance(model_id, str):
        raise VirtualCellContractError("model SafeTensors lacks a model ID")
    return LinearResponseModel(
        model_id=model_id,
        family=family,
        feature_ids=cast(tuple[str, ...], feature_ids),
        gene_ids=cast(tuple[str, ...], gene_ids),
        standardization=standardization,
        coefficients=np.asarray(tensors["coefficients"], dtype=np.float64),
        intercept=np.asarray(tensors["intercept"], dtype=np.float64),
        response_basis=basis,
        admission_required=admission_required,
        admission_feature_count=admission_feature_count,
        prefix_admission_model=prefix_model,
        residual_scale=np.asarray(tensors["residual_scale"], dtype=np.float64),
    )


def _envelope_root_attributes(*, inner_sha256: str) -> dict[str, str]:
    return {
        "empirical_lawhood_clocks": '{"payload":"not-applicable"}',
        "empirical_lawhood_frames": '{"payload":"h5ad-byte-stream"}',
        "empirical_lawhood_keys": '["payload"]',
        "empirical_lawhood_payload_schema": PREDICTION_H5AD_ENVELOPE_SCHEMA,
        "empirical_lawhood_units": '{"payload":"byte"}',
        "inner_media_type": "application/x-hdf5",
        "inner_sha256": inner_sha256,
    }


def write_h5ad_envelope(
    *,
    inner_h5ad_path: Path,
    envelope_path: Path,
    inner_sha256: str,
) -> None:
    """Preserve exact H5AD bytes in a generically auditable HDF5 inventory."""

    if not inner_h5ad_path.is_file() or inner_h5ad_path.is_symlink():
        raise VirtualCellContractError("inner H5AD must be one regular file")
    size = inner_h5ad_path.stat().st_size
    if size <= 0:
        raise VirtualCellContractError("inner H5AD is empty")
    import hashlib

    digest = hashlib.sha256()
    with inner_h5ad_path.open("rb") as source:
        while chunk := source.read(_ENVELOPE_CHUNK_BYTES):
            digest.update(chunk)
    if digest.hexdigest() != inner_sha256:
        raise VirtualCellContractError("inner H5AD hash differs before enveloping")
    with h5py.File(envelope_path, "w") as envelope:
        for key, value in _envelope_root_attributes(inner_sha256=inner_sha256).items():
            envelope.attrs[key] = np.bytes_(value.encode("utf-8"))
        dataset = envelope.create_dataset(
            "payload",
            shape=(size,),
            dtype=np.uint8,
            chunks=(min(size, _ENVELOPE_CHUNK_BYTES),),
        )
        with inner_h5ad_path.open("rb") as source:
            offset = 0
            while chunk := source.read(_ENVELOPE_CHUNK_BYTES):
                dataset[offset : offset + len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
                offset += len(chunk)


def extract_h5ad_envelope(
    *,
    envelope_stream: BinaryIO,
    destination: BinaryIO,
) -> str:
    """Extract and verify the exact inner H5AD without exposing another path."""

    import hashlib

    digest = hashlib.sha256()
    try:
        with closing(h5py.File(envelope_stream, "r")) as envelope:
            expected = envelope.attrs.get("inner_sha256")
            if isinstance(expected, bytes):
                expected = expected.decode("utf-8")
            if not isinstance(expected, str) or "payload" not in envelope:
                raise VirtualCellContractError("H5AD envelope metadata is incomplete")
            payload = envelope["payload"]
            if (
                not isinstance(payload, h5py.Dataset)
                or payload.ndim != 1
                or payload.dtype != np.uint8
            ):
                raise VirtualCellContractError("H5AD envelope payload has the wrong structure")
            for offset in range(0, payload.shape[0], _ENVELOPE_CHUNK_BYTES):
                chunk = np.asarray(
                    payload[offset : offset + _ENVELOPE_CHUNK_BYTES],
                    dtype=np.uint8,
                ).tobytes()
                destination.write(chunk)
                digest.update(chunk)
    except VirtualCellContractError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise VirtualCellContractError("H5AD envelope extraction failed") from error
    observed = digest.hexdigest()
    if observed != expected:
        raise VirtualCellContractError("H5AD envelope inner SHA-256 differs")
    return observed


__all__ = [
    "control_reservoir_arrow_schema",
    "decode_control_reservoir",
    "decode_linear_model",
    "decode_official_metrics",
    "decode_predicted_means",
    "decode_response_summary",
    "decode_target_features",
    "encode_linear_model",
    "encode_official_metrics",
    "encode_control_reservoir",
    "encode_predicted_means",
    "encode_response_summary",
    "encode_target_features",
    "extract_h5ad_envelope",
    "predicted_mean_arrow_schema",
    "official_metrics_arrow_schema",
    "response_summary_arrow_schema",
    "target_feature_arrow_schema",
    "write_h5ad_envelope",
]
