"""Closed causal, private and evaluator records; no outcomes enter selectors."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from .config import CONTEXTS, FrontierDesign, assignment

CAUSAL_FIELDS = ("observations", "requests", "stages", "exposure")
NATIVE_FIELDS = (*CAUSAL_FIELDS, "grid", "callback_cpu")


@dataclass(frozen=True, slots=True)
class FrontierContext(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-context'
    root: str
    context: str
    callback: int | None
    arrays: LocalArrayPayload
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        assignment(self.root)
        if self.context not in CONTEXTS:
            raise ValueError("undeclared frontier context")
        arrays = self.arrays.unpack()
        if self.callback is None:
            if arrays or not self.reasons:
                raise ValueError("no-contact context invented observations")
            return
        low, high = {
            "early": (60, 120),
            "prepared_t0": (60, 600),
            "middle": (720, 900),
            "late": (1440, 1620),
        }[self.context]
        if type(self.callback) is not int or not low <= self.callback <= high:
            raise ValueError("causal callback is outside its fixed context window")
        expected = {
            "observations": (self.callback + 1, 4),
            "requests": (self.callback, 2),
            "stages": (self.callback, 4),
            "exposure": (self.callback, 10, 4),
        }
        if (
            set(arrays) != set(expected)
            or any(
                arrays[k].shape != shape
                or arrays[k].dtype != np.dtype("float64")
                or not np.isfinite(arrays[k]).all()
                for k, shape in expected.items()
            )
            or not np.array_equal(arrays["observations"][:, 0], np.arange(self.callback + 1) * 10)
        ):
            raise ValueError("causal context includes future, private or missing data")


@dataclass(frozen=True, slots=True)
class FrontierPreparation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-preparation'
    root: str
    recipe: ObjectIdentity
    source_sha256: str
    contexts: tuple[FrontierContext, ...]
    native_calls: int
    retained_parents: tuple[ObjectIdentity, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        role, _ = assignment(self.root)
        validate_sha256(self.source_sha256, field_name="source_sha256")
        design = FrontierDesign()
        if (
            self.recipe != ObjectIdentity.from_record(design.config_id, design)
            or tuple(c.context for c in self.contexts) != CONTEXTS
            or any(c.root != self.root for c in self.contexts)
            or type(self.native_calls) is not int
            or not 0 <= self.native_calls <= 2
            or (
                role == "development"
                and (self.native_calls != 0 or len(self.retained_parents) != 2)
            )
            or (role != "development" and self.retained_parents)
        ):
            raise ValueError("frontier preparation changed assignment or retention lineage")

    @property
    def record_id(self) -> str:
        return f"{self.root}.frontier-preparation"


@dataclass(frozen=True, slots=True)
class FrontierPrivate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-private'
    root: str
    preparation: ObjectIdentity
    arrays: LocalArrayPayload
    failures: tuple[str | None, ...]

    def __post_init__(self) -> None:
        assignment(self.root)
        if (
            self.preparation.object_schema != FrontierPreparation.SCHEMA
            or len(self.failures) not in (0, 1, 2)
            or set(self.arrays.unpack())
            != {f"v{v}_{field}" for v in range(len(self.failures)) for field in NATIVE_FIELDS}
        ):
            raise ValueError("private preparation has a different parent/view census")


@dataclass(frozen=True, slots=True)
class FrontierAssay(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-assay'
    root: str
    preparation: ObjectIdentity
    seal: ObjectIdentity
    arrays: LocalArrayPayload
    native_calls: int
    failures: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        assignment(self.root)
        if (
            self.preparation.object_schema != FrontierPreparation.SCHEMA
            or self.seal.object_schema != 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-prediction-seal'
            or type(self.native_calls) is not int
            or not 0 <= self.native_calls <= 80
        ):
            raise ValueError("frontier assay lost source, seal or native census")


@dataclass(frozen=True, slots=True)
class FrontierRetainedRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-retained-root'
    preparation: FrontierPreparation
    private: FrontierPrivate

    def __post_init__(self) -> None:
        if (
            assignment(self.preparation.root)[0] != "development"
            or self.private.root != self.preparation.root
            or self.private.preparation
            != ObjectIdentity.from_record(self.preparation.record_id, self.preparation)
        ):
            raise ValueError("retained source root lost its exact compatibility binding")
