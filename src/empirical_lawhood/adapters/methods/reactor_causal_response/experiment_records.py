"""Strict, bounded publications between the declared empirical study stages."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import base64
import json
from typing import ClassVar
import zipfile
import numpy as np
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.kernel.provenance import ObjectIdentity
from .payload import DevelopmentBoundReactorResponsePayload
from .terminal import EmpiricalQualificationResult


@dataclass(frozen=True, slots=True)
class EmpiricalArrayPayload(CanonicalRecord):
    """Numeric arrays only; missing arrays remain explicit empty arrays."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-causal-response/empirical-array-payload'
    sha256: str
    npz_base64: str

    def __post_init__(self) -> None:
        validate_sha256(self.sha256, field_name="sha256")
        if len(self.npz_base64) > 128 * 1024**2:
            raise ValueError("empirical array publication exceeds its byte bound")

    @classmethod
    def pack(cls, arrays: dict[str, np.ndarray]) -> 'EmpiricalArrayPayload':
        if any(a.dtype.kind not in "fiu" for a in arrays.values()):
            raise ValueError("only numeric scientific array operands are allowed")
        out = BytesIO()
        np.savez_compressed(out, allow_pickle=False, **arrays)
        raw = out.getvalue()
        return cls(sha256(raw).hexdigest(), base64.b64encode(raw).decode())

    def unpack(self) -> dict[str, np.ndarray]:
        raw = base64.b64decode(self.npz_base64, validate=True)
        if sha256(raw).hexdigest() != self.sha256:
            raise ValueError("empirical array digest differs")
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            if (
                len(archive.infolist()) > 128
                or sum(i.file_size for i in archive.infolist()) > 256 * 1024**2
            ):
                raise ValueError("empirical decompression exceeds its bound")
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if any(a.dtype.kind not in "fiu" for a in arrays.values()):
            raise ValueError("non-numeric empirical array")
        return arrays


@dataclass(frozen=True, slots=True)
class EmpiricalEpisodeEnvelope(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-causal-response/empirical-episode-envelope'
    root: str
    episode: str
    refined: bool
    failure: str | None
    arrays: EmpiricalArrayPayload

    def __post_init__(self) -> None:
        validate_stable_id(self.root, field_name="root")
        if (
            self.episode
            not in (
                "exploration-unshifted",
                "exploration-shifted-one-position",
                "exploration-shifted-two-positions",
                "feed-intervention",
                "jacket-intervention",
                "EL",
                "F0",
                "F1",
                "MARGIN",
                "FIXED",
                "REF",
                "EKF",
                "SCHEDULED_BACKOFF_ZERO",
                "SCHEDULED_BACKOFF_HALF",
                "calibration",
                "qualification",
            )
            or type(self.refined) is not bool
        ):
            raise ValueError("undeclared episode/view")
        if self.failure is not None:
            validate_stable_id(self.failure.lower().replace("_", "-"), field_name="failure")


@dataclass(frozen=True, slots=True)
class EmpiricalAcquisitionEnvelope(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-causal-response/empirical-acquisition-envelope'
    root: str
    role: str
    episodes: tuple[EmpiricalEpisodeEnvelope, ...]

    def __post_init__(self) -> None:
        from empirical_lawhood.adapters.simulators.reactor_causal_response.design import ROLES

        count = next((n for role, n, _ in ROLES if role == self.role), 0)
        if self.root not in {f"reactor-empirical-{self.role}-{i:03d}" for i in range(count)}:
            raise ValueError("acquisition root is outside its frozen role")
        if any(e.root != self.root for e in self.episodes) or len(
            {(e.episode, e.refined) for e in self.episodes}
        ) != len(self.episodes):
            raise ValueError("substituted/duplicate nested episode")
        expected = (
            tuple((name, view) for name in ("exploration-unshifted", "exploration-shifted-one-position", "exploration-shifted-two-positions", "feed-intervention", "jacket-intervention") for view in (False, True))
            if self.role in ("fit", "nomination")
            else ((self.role, False), (self.role, True))
        )
        if self.role == "confirmation":
            from .config import ARMS

            rotation = int(self.root.rsplit("-", 1)[1]) % len(ARMS)
            order = ARMS[rotation:] + ARMS[:rotation]
            expected = tuple((arm, False) for arm in order) + (("EL", True),)
        if tuple((e.episode, e.refined) for e in self.episodes) != expected:
            raise ValueError("missing assigned acquisition slot")


@dataclass(frozen=True, slots=True)
class EmpiricalDiscoveryEnvelope(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-causal-response/empirical-discovery-envelope'
    recipe: ObjectIdentity
    acquisitions: tuple[ObjectIdentity, ...]
    result_json: str
    rows: EmpiricalArrayPayload | None = None

    def __post_init__(self) -> None:
        if len(self.acquisitions) != 24 or len(self.result_json.encode()) > 16 * 1024**2:
            raise ValueError("discovery publication census/size differs")
        value = json.loads(self.result_json)
        if not isinstance(value, dict) or value.get("disposition") not in (
            "NOMINATED",
            "IDENTIFICATION_NOT_SUPPORTED",
        ):
            raise ValueError("invalid discovery terminal")


@dataclass(frozen=True, slots=True)
class EmpiricalCalibrationEnvelope(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-causal-response/empirical-calibration-envelope'
    discovery: ObjectIdentity
    roots: tuple[ObjectIdentity, ...]
    payload: DevelopmentBoundReactorResponsePayload | None
    result_json: str

    def __post_init__(self) -> None:
        expected = tuple(f"reactor-empirical-calibration-{i:03d}" for i in range(32))
        if (
            tuple(r.object_id for r in self.roots) != expected
            or len(self.result_json.encode()) > 65536
        ):
            raise ValueError("calibration publication census/size differs")
        result = json.loads(self.result_json)
        if result.get("disposition") not in ("CALIBRATED", "CALIBRATION_UNUSABLE"):
            raise ValueError("invalid calibration terminal")
        q = result["q"]
        frozen_q = None if self.payload is None or self.payload.q is None else float(self.payload.q)
        if (None if q is None or q > 1 else float(q)) != frozen_q:
            raise ValueError("frozen calibration payload differs from rank result")


@dataclass(frozen=True, slots=True)
class EmpiricalUsefulness(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-causal-response/empirical-usefulness'
    roots: tuple[tuple[str, int, int], ...]

    def __post_init__(self) -> None:
        if tuple(r for r, _, _ in self.roots) != tuple(
            f"reactor-empirical-qualification-{i:03d}" for i in range(32)
        ):
            raise ValueError("usefulness must retain the complete assigned qualification panel")
        if any(
            not 0 <= alternatives <= completed <= 2880 for _, completed, alternatives in self.roots
        ):
            raise ValueError("invalid usefulness decision counts")

    @property
    def useful(self) -> bool:
        return all(n == 2880 for _, n, _ in self.roots) and any(a > 0 for _, _, a in self.roots)


@dataclass(frozen=True, slots=True)
class EmpiricalQualificationTerminal(CanonicalRecord):
    """Distinguish a prerequisite nonentry from an entered owner's law result."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-causal-response/empirical-qualification-terminal'
    recipe: ObjectIdentity
    calibration: ObjectIdentity
    roots: tuple[ObjectIdentity, ...]
    result: EmpiricalQualificationResult | None
    prerequisite: str | None
    usefulness: EmpiricalUsefulness | None = None

    def __post_init__(self) -> None:
        expected = tuple(
            f"reactor-empirical-{role}-{i:03d}"
            for role in ("calibration", "qualification")
            for i in range(32)
        )
        if tuple(r.object_id for r in self.roots) != expected:
            raise ValueError("qualification terminal lost assigned root identities")
        if self.prerequisite not in (None, "IDENTIFICATION_NOT_SUPPORTED", "CALIBRATION_UNUSABLE"):
            raise ValueError("undeclared qualification prerequisite")
        if (self.result is None) != (self.prerequisite is not None):
            raise ValueError("qualification nonentry must not manufacture a law result")
        if self.result is not None and self.result.design != self.recipe:
            raise ValueError("qualification owner result changed recipe")
        if (self.result is None) != (self.usefulness is None):
            raise ValueError("entered qualification must retain its separate usefulness readout")
