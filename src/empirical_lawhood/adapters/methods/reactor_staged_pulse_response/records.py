"Causal and sealed native operands with exact custody schemas."

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.reactor_causal_response.experiment_records import EmpiricalEpisodeEnvelope
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorAssignedScenario
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from .config import CONTEXTS, FIRST, assignment

CAUSAL_FIELDS = ("observations", "requests", "stages", "exposure")


@dataclass(frozen=True, slots=True)
class ClassicalContext(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-context'
    root: str
    context: str
    callback: int | None
    arrays: LocalArrayPayload
    reasons: tuple[str, ...]
    first_callback: int | None = None
    predecessor: ObjectIdentity | None = None
    owner_scope: str | None = None

    def __post_init__(self) -> None:
        assignment(self.root)
        if self.owner_scope is not None:
            validate_stable_id(self.owner_scope, field_name="owner_scope")
        if (self.predecessor is None) != (self.owner_scope is None):
            raise ValueError("prospective induced history requires its distinct owner scope")
        if self.context not in (*CONTEXTS, "induced"):
            raise ValueError("undeclared staged-pulse causal context")
        values = self.arrays.unpack()
        if self.callback is None:
            if values or not self.reasons or self.predecessor is not None:
                raise ValueError("no-contact context invented causal inputs")
            return
        if type(self.callback) is not int or not 60 <= self.callback <= 600:
            raise ValueError("Staged-pulse causal clock is outside its declared local window")
        if self.context == "early" and self.callback > 120:
            raise ValueError("early context is outside its declared first-contact window")
        expected = {
            "observations": (self.callback + 1, 4),
            "requests": (self.callback, 2),
            "stages": (self.callback, 4),
            "exposure": (self.callback, 10, 4),
        }
        if (
            set(values) != set(expected)
            or any(
                values[k].shape != shape
                or values[k].dtype != np.dtype("float64")
                or not np.isfinite(values[k]).all()
                for k, shape in expected.items()
            )
            or not np.array_equal(values["observations"][:, 0], np.arange(self.callback + 1) * 10)
        ):
            raise ValueError(
                "causal context includes private/future data or loses its native prefix"
            )
        if self.context == "induced":
            k = self.first_callback
            if k is None or not 60 <= k <= 120 or self.callback != k + 12:
                raise ValueError("induced context lacks the exact first-induced 120-second history")
            jacket = values["stages"][k - 1, 3]
            if (
                not np.array_equal(values["requests"][k:, 0], [float(r) for r in FIRST.rates()])
                or not (values["requests"][k:, 1] == jacket).all()
                or not (values["stages"][k:, 3] == jacket).all()
                or values["stages"][-1, 2] != 0
            ):
                raise ValueError("induced context substitutes the initial prefix, another first word or jacket")
            if (
                self.predecessor is not None
                and self.predecessor.object_schema != 'empirical-lawhood/runtime/delivery-controller-tick-receipt'
            ):
                raise ValueError("induced prospective history requires an actual first owner tick")
        elif self.first_callback is not None or self.predecessor is not None:
            raise ValueError("ordinary context has an undeclared prospective predecessor")

    @property
    def record_id(self) -> str:
        scope = "" if self.owner_scope is None else f".{self.owner_scope}"
        return f"{self.root}{scope}.{self.context}.causal-capture"


@dataclass(frozen=True, slots=True)
class ClassicalPreparation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-preparation'
    root: str
    block: str
    source_sha256: str
    contexts: tuple[ClassicalContext, ...]
    native_calls: int
    retained_parents: tuple[ObjectIdentity, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        assignment(self.root)
        validate_sha256(self.source_sha256, field_name="source_sha256")
        expected = ("early",) if self.block == "staged-sequence-comparison" else CONTEXTS
        if (
            self.block not in ("base-menu-comparison", "expanded-menu-comparison", "staged-sequence-comparison")
            or tuple(c.context for c in self.contexts) != expected
            or any(c.root != self.root for c in self.contexts)
            or type(self.native_calls) is not int
            or not 0 <= self.native_calls <= 2
        ):
            raise ValueError(
                "preparation changes its context, independent root or acquisition census"
            )
        historical = assignment(self.root)[1] == "development"
        if historical and (self.native_calls or len(self.retained_parents) != 2):
            raise ValueError("historical preparation must reuse its exact causal/private parents")

    @property
    def record_id(self) -> str:
        return f"{self.root}.preparation"


@dataclass(frozen=True, slots=True)
class ClassicalPrivate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-private'
    preparation: ObjectIdentity
    scenario: ReactorAssignedScenario
    episodes: tuple[EmpiricalEpisodeEnvelope, ...]

    def __post_init__(self) -> None:
        block, role, seed = assignment(self.scenario.unit_id)
        if (
            self.preparation.object_schema != ClassicalPreparation.SCHEMA
            or self.scenario.seed != seed
            or self.scenario.split != ("calibration" if role == "development" else "heldout")
            or len(self.episodes) > 2
            or any(e.root != self.scenario.unit_id for e in self.episodes)
        ):
            raise ValueError("private preparation changes source/scenario or paired-view identity")


@dataclass(frozen=True, slots=True)
class ClassicalNativeWindow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-native-window'
    root: str
    branch_id: str
    refined: bool
    start_s: D
    duration_s: int
    requested_rates: tuple[D, ...]
    arrays: LocalArrayPayload
    failure: str | None

    def __post_init__(self) -> None:
        assignment(self.root)
        validate_stable_id(self.branch_id, field_name="branch_id")
        if (
            self.duration_s not in (120, 240)
            or len(self.requested_rates) != self.duration_s // 10
            or type(self.refined) is not bool
        ):
            raise ValueError("native window changes its declared guard or requested word")
        if not self.arrays.unpack() and self.failure is None:
            raise ValueError("empty native measurement requires its explicit missing role")


@dataclass(frozen=True, slots=True)
class ClassicalAssay(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-assay'
    root: str
    preparation: ObjectIdentity
    seal: ObjectIdentity
    windows: tuple[ClassicalNativeWindow, ...]
    induced: tuple[ClassicalContext, ...]
    native_calls: int
    reasons: tuple[str, ...]
    induced_seal: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        assignment(self.root)
        if (
            self.preparation.object_schema != ClassicalPreparation.SCHEMA
            or type(self.native_calls) is not int
            or not 0 <= self.native_calls <= 52
            or any(w.root != self.root for w in self.windows)
            or len({(w.branch_id, w.refined) for w in self.windows}) != len(self.windows)
            or any(c.root != self.root or c.context != "induced" for c in self.induced)
        ):
            raise ValueError("assay changed its source/decision/view census")
        if bool(self.induced) != (self.induced_seal is not None) or (
            self.induced_seal is not None
            and self.induced_seal.object_schema
            != 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-induced-prediction'
        ):
            raise ValueError(
                "conditional measurement lacks its pre-second-response prediction seal"
            )

    @property
    def record_id(self) -> str:
        return f"{self.root}.assay"
