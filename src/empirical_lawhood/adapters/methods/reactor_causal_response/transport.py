"""Bounded canonical raw-root transport, preserving every exposure substep."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import base64
from io import BytesIO
from typing import ClassVar
import zipfile
import numpy as np
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from .calibration import RootOperands


@dataclass(frozen=True, slots=True)
class NumericReactorRootEnvelope(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reactor-response/numeric-reactor-root-envelope'
    ARRAY_FIELDS: ClassVar[frozenset[str]] = frozenset(
        set(RootOperands.__dataclass_fields__)
        - {"root", "observations", "consumer_q", "model_sha256", "stages"}
    )
    root: str
    content_sha256: str
    npz_base64: str

    def __post_init__(self) -> None:
        validate_stable_id(self.root, field_name="root")
        validate_sha256(self.content_sha256, field_name="content_sha256")
        if len(self.npz_base64) > 32 * 1024**2:
            raise ValueError("root transport exceeds limit")

    @classmethod
    def pack(cls, raw: RootOperands) -> 'NumericReactorRootEnvelope':
        out = BytesIO()
        np.savez_compressed(out, **{k: v for k, v in asdict(raw).items() if k in cls.ARRAY_FIELDS})
        data = out.getvalue()
        return cls(raw.root, sha256(data).hexdigest(), base64.b64encode(data).decode())

    def unpack(self) -> RootOperands:
        data = base64.b64decode(self.npz_base64, validate=True)
        if sha256(data).hexdigest() != self.content_sha256:
            raise ValueError("raw root transport digest differs")
        with zipfile.ZipFile(BytesIO(data)) as archive:
            if sum(i.file_size for i in archive.infolist()) > 32 * 1024**2:
                raise ValueError("root decompression exceeds bound")
        with np.load(BytesIO(data), allow_pickle=False) as raw:
            keys = self.ARRAY_FIELDS
            if set(raw.files) != keys:
                raise ValueError("raw root fields differ")
            arrays = {k: raw[k] for k in keys}
            if any(v.dtype.kind not in "fi" for v in arrays.values()):
                raise ValueError("non-numeric root operand")
            return RootOperands(self.root, **arrays)


@dataclass(frozen=True, slots=True)
class CausalReactorRootEnvelope(NumericReactorRootEnvelope):
    "Policy- and observation-bound causal-response transport; earlier records cannot supply this evidence."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reactor-response/causal-reactor-root-envelope'
    VERSION: ClassVar[str] = '1.0.0'
    ARRAY_FIELDS: ClassVar[frozenset[str]] = frozenset(
        set(RootOperands.__dataclass_fields__) - {"root"}
    )
    nonattempt_reason: str | None = None

    def __post_init__(self) -> None:
        NumericReactorRootEnvelope.__post_init__(self)
        if self.nonattempt_reason not in (
            None,
            "IDENTIFICATION_NOT_SUPPORTED",
            "CALIBRATION_UNUSABLE",
        ):
            raise ValueError("undeclared empirical prerequisite stop")
        if self.nonattempt_reason is not None:
            raw = self.unpack()
            if any(getattr(raw, name).size for name in self.ARRAY_FIELDS):
                raise ValueError(
                    "unentered prerequisite cannot contain acquired or model-derived rows"
                )

    @classmethod
    def nonattempt(cls, root: str, reason: str) -> 'CausalReactorRootEnvelope':
        raw = RootOperands(
            root,
            np.empty((0,)),
            np.empty((0,), dtype=int),
            np.empty((0, 23)),
            np.empty((0, 2, 3)),
            np.empty((0, 10, 4)),
            np.empty((0, 10, 4)),
            np.empty((0, 20, 4)),
            np.empty((0, 20, 4)),
            np.empty((0, 2, 2)),
        )
        envelope = cls.pack(raw)
        return cls(envelope.root, envelope.content_sha256, envelope.npz_base64, reason)
