"""Strict frozen local discovery payload and assigned fresh measurement records."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.methods.reactor_causal_response.experiment_records import EmpiricalArrayPayload
from .discovery import CONTEXT


@dataclass(frozen=True, slots=True)
class LocalArrayPayload(EmpiricalArrayPayload):
    """Lossless local-assay arrays with the declared 1,365-array ceiling.

    The old development tensor codec has a 128-array census, which does not
    describe 93 action slots, two views and per-branch custody fields.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-local-domain-qualification/local-array-payload'

    @classmethod
    def pack(cls, arrays: dict[str, np.ndarray]) -> 'LocalArrayPayload':
        packed = EmpiricalArrayPayload.pack(arrays)
        return cls(packed.sha256, packed.npz_base64)

    def unpack(self) -> dict[str, np.ndarray]:
        import base64
        from hashlib import sha256
        from io import BytesIO
        import zipfile

        raw = base64.b64decode(self.npz_base64, validate=True)
        if sha256(raw).hexdigest() != self.sha256:
            raise ValueError("local array digest differs")
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            entries = archive.infolist()
            if (
                len(entries) > 1365
                or len({i.filename for i in entries}) != len(entries)
                or any(
                    not i.filename.endswith(".npy") or "/" in i.filename or "\\" in i.filename
                    for i in entries
                )
                or sum(i.file_size for i in entries) > 256 * 1024**2
            ):
                raise ValueError("local array expansion exceeds its exact assay bound")
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if any(a.dtype.kind not in "fiu" for a in arrays.values()):
            raise ValueError("non-numeric local array")
        return arrays


@dataclass(frozen=True, slots=True)
class LocalDomain(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-local-domain-qualification/local-domain'
    domain_id: str
    path: tuple[tuple[int, D, bool], ...]
    mean: tuple[D, ...]
    scale: tuple[D, ...]
    operator: tuple[tuple[D, D], ...]
    support_lower: tuple[D, ...]
    support_upper: tuple[D, ...]
    action_fit_roots: tuple[int, ...]
    nominated_receivers: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.domain_id, field_name="domain_id")
        if (
            not 1 <= len(self.domain_id) <= 8
            or len(self.path) > 6
            or any(
                c not in CONTEXT or not t.is_finite() or type(left) is not bool
                for c, t, left in self.path
            )
            or len(self.mean) != 23
            or len(self.scale) != 23
            or len(self.operator) != 24
            or any(len(row) != 2 for row in self.operator)
            or len(self.support_lower) != len(CONTEXT)
            or len(self.support_upper) != len(CONTEXT)
            or len(self.action_fit_roots) != 9
            or any(type(n) is not int or not 0 <= n <= 16 for n in self.action_fit_roots)
            or self.nominated_receivers not in ((), (0,), (1,), (0, 1))
        ):
            raise ValueError("invalid frozen local-domain axes")
        values = (
            *self.mean,
            *self.scale,
            *self.support_lower,
            *self.support_upper,
            *(v for row in self.operator for v in row),
        )
        if (
            any(not v.is_finite() for v in values)
            or any(v <= 0 for v in self.scale)
            or any(lo > hi for lo, hi in zip(self.support_lower, self.support_upper, strict=True))
        ):
            raise ValueError("invalid frozen local-domain numerical operands")

    @property
    def actions(self) -> tuple[int, ...]:
        return tuple(i for i, n in enumerate(self.action_fit_roots) if n >= 8)

    def contains(self, x: np.ndarray) -> np.ndarray:
        mask = np.ones(len(x), dtype=bool)
        for coordinate, threshold, left in self.path:
            mask &= (
                (x[:, coordinate] <= float(threshold))
                if left
                else (x[:, coordinate] > float(threshold))
            )
        return mask

    def proposed_support(self, x: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return (
            self.contains(x)
            & (
                (x[:, CONTEXT] >= np.array(self.support_lower, dtype=float))
                & (x[:, CONTEXT] <= np.array(self.support_upper, dtype=float))
            ).all(axis=1)
            & np.isin(actions, self.actions)
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - np.asarray(self.mean, dtype=float)) / np.asarray(self.scale, dtype=float)
        operator = np.asarray(self.operator, dtype=float)
        return np.asarray(z @ operator[:-1] + operator[-1])


@dataclass(frozen=True, slots=True)
class LocalAtlas(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-local-domain-qualification/local-atlas'
    atlas_id: str
    development_sha256: str
    domains: tuple[LocalDomain, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.atlas_id, field_name="atlas_id")
        if len(self.development_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.development_sha256
        ):
            raise ValueError("invalid discovery provenance")
        ids = tuple(d.domain_id for d in self.domains)
        if not 1 <= len(ids) <= 16 or len(set(ids)) != len(ids):
            raise ValueError("invalid finite local-law collection")
        # Check the complete binary partition structurally, without test queries.
        paths = tuple(d.path for d in self.domains)

        def partition(branches: tuple[tuple[tuple[int, D, bool], ...], ...]) -> bool:
            if not branches:
                return False
            if any(not p for p in branches):
                return branches == ((),)
            splits = {(p[0][0], p[0][1]) for p in branches}
            return len(splits) == 1 and all(
                partition(tuple(p[1:] for p in branches if p[0][2] is side))
                for side in (True, False)
            )

        if not partition(paths):
            raise ValueError("local discovery domains overlap or leave an undeclared partition gap")

    @property
    def candidates(self) -> tuple[LocalDomain, ...]:
        return tuple(d for d in self.domains if d.nominated_receivers)


@dataclass(frozen=True, slots=True)
class LocalRootEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-local-domain-qualification/local-root-evidence'
    root: str
    role: str
    seed: int
    recipe: ObjectIdentity
    # Lossless arrays: complete donor trajectories and branch measurement windows.
    arrays: LocalArrayPayload
    # domain, donor policy, callback, action; None marks no causal contact.
    assays: tuple[tuple[str, int | None, int | None, tuple[int, ...]], ...]
    failures: tuple[tuple[str, str], ...]
    native_calls: int

    def __post_init__(self) -> None:
        if self.role not in ("calibration", "qualification") or self.root not in {
            f"reactor-local-domain-{self.role}-{i:03d}" for i in range(32)
        }:
            raise ValueError("root is outside the frozen fresh assignment")
        index = int(self.root.rsplit("-", 1)[1])
        if self.seed != (93000 if self.role == "calibration" else 94000) + index:
            raise ValueError("fresh root seed differs")
        if not 0 <= self.native_calls <= 192 or len(self.assays) > 16:
            raise ValueError("local source allocation exceeded")
        if len({d for d, _, _, _ in self.assays}) != len(self.assays):
            raise ValueError("duplicate local assay")
        for domain, policy, callback, actions in self.assays:
            validate_stable_id(domain, field_name="assay_domain")
            if (
                (policy is None) != (callback is None)
                or (policy is None and actions)
                or (
                    policy is not None
                    and (policy not in (0, 1, 2) or callback is None or not 0 <= callback < 2880)
                )
                or tuple(sorted(set(actions))) != actions
                or any(a not in range(9) for a in actions)
            ):
                raise ValueError("invalid declared causal assay anchor or action census")
