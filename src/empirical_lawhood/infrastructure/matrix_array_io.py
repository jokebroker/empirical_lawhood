"""Bounded no-pickle array framing for public reusable matrix operands."""

from collections.abc import Mapping
from hashlib import sha256
from io import BytesIO
from types import MappingProxyType
import zipfile

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.kernel.matrix_inputs import (
    MAXIMUM_MATRIX_ARRAY_BYTES,
    MatrixArrayMember,
    SavedMatrixArrays,
)


def encode_matrix_arrays(
    arrays: Mapping[str, NDArray[np.generic]],
    *,
    nonfinite_members: tuple[str, ...] = (),
) -> tuple[bytes, tuple[MatrixArrayMember, ...]]:
    """Retain exact typed values; missingness must be explicitly declared."""
    if not 0 < len(arrays) <= 256 or not set(nonfinite_members) <= set(arrays):
        raise ValueError("Matrix array member or missingness census differs")
    members = []
    total = 0
    for name, value in sorted(arrays.items()):
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise ValueError("Matrix array operands refuse Python object serialization")
        member = MatrixArrayMember(name, value.dtype.str, value.shape, sha256(value.tobytes(order="C")).hexdigest(), name not in nonfinite_members)
        total += member.size_bytes
        if total > MAXIMUM_MATRIX_ARRAY_BYTES or member.finite_required and not np.isfinite(value).all():
            raise ValueError("Matrix array expansion exceeds its bound or complete population")
        members.append(member)
    stream = BytesIO()
    np.savez_compressed(stream, **{name: np.array(value, order="C", copy=True) for name, value in arrays.items()})
    raw = stream.getvalue()
    if len(raw) > MAXIMUM_MATRIX_ARRAY_BYTES:
        raise ValueError("Matrix array physical transport exceeds its byte bound")
    return raw, tuple(members)


def decode_matrix_arrays(
    raw: bytes, manifest: SavedMatrixArrays
) -> Mapping[str, NDArray[np.generic]]:
    """Authenticate physical bytes and bounded NPY headers before expansion."""
    if (
        type(raw) is not bytes
        or len(raw) > MAXIMUM_MATRIX_ARRAY_BYTES
        or len(raw) != manifest.array_artifact.size_bytes
        or sha256(raw).hexdigest() != manifest.array_artifact.sha256
    ):
        raise ValueError("Matrix array transport differs from its current artifact")
    declared = {f"{member.name}.npy": member for member in manifest.members}
    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            infos = archive.infolist()
            if (
                len(infos) != len(declared)
                or {info.filename for info in infos} != set(declared)
                or sum(info.file_size for info in infos) > MAXIMUM_MATRIX_ARRAY_BYTES + 256 * 10000
            ):
                raise ValueError("Matrix archive members or expanded byte census differs")
            for info in infos:
                member = declared[info.filename]
                if info.file_size > member.size_bytes + 10000 or info.flag_bits & 1:
                    raise ValueError("Matrix archive member is oversized or encrypted")
                with archive.open(info) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10000)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=10000)
                    else:
                        raise ValueError("Unsupported bounded matrix NPY header revision")
                    if shape != member.shape or dtype.str != member.dtype or dtype.hasobject or fortran:
                        raise ValueError("Matrix NPY axes, dtype or layout differ from the manifest")
                    if stream.tell() + member.size_bytes != info.file_size:
                        raise ValueError("Matrix NPY payload length differs from the declared axes")
            values = {}
            with np.load(BytesIO(raw), allow_pickle=False, max_header_size=10000) as payload:
                for member in manifest.members:
                    value = payload[member.name]
                    if sha256(value.tobytes(order="C")).hexdigest() != member.sha256 or member.finite_required and not np.isfinite(value).all():
                        raise ValueError("Matrix array values or required finite population differ")
                    immutable = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
                    values[member.name] = immutable
        return MappingProxyType(values)
    except (OSError, zipfile.BadZipFile, EOFError) as error:
        raise ValueError("Matrix array transport cannot be decoded within its member contract") from error
