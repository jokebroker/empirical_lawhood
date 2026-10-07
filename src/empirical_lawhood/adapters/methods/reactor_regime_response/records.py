"Versioned, role-separated custody for the reactor regime-response study native panel."

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .config import ROOTS


def _assignment(root: str, role: str, seed: int) -> None:
    if (root, role, seed) not in {(r, p, s) for r, p, _, s in ROOTS}:
        raise ValueError("reactor regime-response study root is outside the exact assignment")


@dataclass(frozen=True, slots=True)
class RegimeCausalPreparation(CanonicalRecord):
    """Only observed/action inputs; the downstream predictor gets no native grid."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-causal-preparation'
    root: str
    role: str
    seed: int
    recipe: ObjectIdentity
    source_sha256: str
    anchors: tuple[tuple[str, int | None, str], ...]
    arrays: LocalArrayPayload
    native_calls: int
    failures: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _assignment(self.root, self.role, self.seed)
        if (
            tuple(a[0] for a in self.anchors)
            != ("early", "middle", "late", "prepared_t0")
            or not 1 <= self.native_calls <= 6
        ):
            raise ValueError("reactor causal preparation census differs")


@dataclass(frozen=True, slots=True)
class RegimePrivatePreparation(CanonicalRecord):
    """Evaluator-only native grids and call clocks for the same preparation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-private-preparation'
    root: str
    role: str
    seed: int
    causal_preparation: ObjectIdentity
    arrays: LocalArrayPayload
    native_calls: int
    failures: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _assignment(self.root, self.role, self.seed)
        if not 1 <= self.native_calls <= 6:
            raise ValueError("reactor private preparation call census differs")


@dataclass(frozen=True, slots=True)
class RegimePredictionSeal(CanonicalRecord):
    """Persisted causal/prediction barrier before any root's action assays.

    Fit roots carry an explicit acquisition-only seal; other roles must name
    their already frozen prediction package.  The provider verifies this
    record's task receipt and content hash before contacting the assay source.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-prediction-seal'

    root: str
    role: str
    seed: int
    causal_preparation: ObjectIdentity
    model_package: ObjectIdentity | None
    context_sha256: tuple[str | None, ...]
    candidate_ids: tuple[str, ...]
    predictions: LocalArrayPayload
    seal_kind: str

    def __post_init__(self) -> None:
        _assignment(self.root, self.role, self.seed)
        if self.seal_kind not in ("FIT_ACQUISITION_ONLY", "FROZEN_PREDICTIONS"):
            raise ValueError("reactor prediction seal kind differs")
        if len(self.context_sha256) != 8 or any(
            value is not None and (len(value) != 64 or any(character not in "0123456789abcdef" for character in value))
            for value in self.context_sha256
        ):
            raise ValueError("reactor seal lacks eight causal context hashes")
        arrays = self.predictions.unpack()
        if self.role == "fit":
            if self.seal_kind != "FIT_ACQUISITION_ONLY" or self.model_package is not None or self.candidate_ids or arrays:
                raise ValueError("fit assay requires only its causal acquisition seal")
        elif (
            self.seal_kind != "FROZEN_PREDICTIONS"
            or self.model_package is None
            or not self.candidate_ids
            or tuple(sorted(set(self.candidate_ids))) != self.candidate_ids
            or set(arrays) != {
                "forecast", "support", "present", "masses", "projection_valid", "context_callback"
            }
            or arrays["forecast"].shape != (len(self.candidate_ids), 8)
            or arrays["support"].shape != (len(self.candidate_ids), 8)
            or arrays["present"].shape != (len(self.candidate_ids), 8)
            or arrays["masses"].shape != (8, 3)
            or arrays["projection_valid"].shape != (8, 3)
            or arrays["context_callback"].shape != (8,)
            or arrays["context_callback"].dtype.kind not in "iu"
            or np.any((arrays["context_callback"] < -1) | (arrays["context_callback"] >= 2880))
            or not np.isfinite(arrays["forecast"]).all()
            or not np.isfinite(arrays["masses"]).all()
            or not np.isin(arrays["support"], (0, 1)).all()
            or not np.isin(arrays["present"], (0, 1)).all()
            or not np.isin(arrays["projection_valid"], (0, 1)).all()
            or np.any(arrays["support"] > arrays["present"])
            or np.any((arrays["present"] == 0) & (arrays["forecast"] != 0))
            or np.any(
                (arrays["context_callback"] == -1)[None, :]
                & (arrays["present"] != 0)
            )
            or any(
                (callback == -1) != (digest is None)
                for callback, digest in zip(arrays["context_callback"], self.context_sha256, strict=True)
            )
        ):
            raise ValueError("nonfit assay lacks a frozen prediction package/census")


@dataclass(frozen=True, slots=True)
class RegimeAssayPanel(CanonicalRecord):
    """All prescribed measured branches after a persisted prediction barrier."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-assay-panel'
    root: str
    role: str
    seed: int
    causal_preparation: ObjectIdentity
    private_preparation: ObjectIdentity
    sealed_prediction: ObjectIdentity
    slots: tuple[tuple[str, int | None, int, int, str], ...]
    arrays: LocalArrayPayload
    native_calls: int
    failures: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _assignment(self.root, self.role, self.seed)
        if (
            len(self.slots) != 48
            or not 0 <= self.native_calls <= 48
            or len({(name, view, word) for name, _, view, word, _ in self.slots}) != 48
        ):
            raise ValueError("reactor assay panel lost a prescribed slot")
