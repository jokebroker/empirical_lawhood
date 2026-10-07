"Fresh calibration census, distinct from the canary and retained preparations.\n\nAll assignments are determined before observations. This module neither contacts\nthe source nor authorizes execution. Native mechanics remain with the prepared-response owner.\n"

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.methods.finite_response_law.science import PARENTS, PROGRAMME, root_seed, seed_for
from .contracts import FiniteResponseLawNativeConfig, FiniteResponseLawNativeInvocation, FiniteResponseLawNativeRoot


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationRoot(FiniteResponseLawNativeRoot):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-calibration-root'
    VERSION: ClassVar[str] = '1.0.0'

    def __post_init__(self) -> None:
        if (
            self.cohort != "calibration"
            or type(self.index) is not int
            or not 0 <= self.index < 32
            or self.retained_root is not None
            or self.seed_sha256
            != sha256(str(seed_for("prefix", self.stage_unit)).encode()).hexdigest()
        ):
            raise ValueError("Finite response-law calibration requires its exact fresh independent root")

    @property
    def stage_unit(self) -> str:
        return f"calibration.r{self.index:03d}"

    @property
    def root_id(self) -> str:
        return f"{PROGRAMME}.calibration.prepared.r{self.index:03d}"

    @property
    def assigned_parent(self) -> str:
        rng = np.random.Generator(np.random.PCG64(root_seed(self, "parent-allocation")))
        return PARENTS[int(rng.integers(0, len(PARENTS)))]


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationConfig(FiniteResponseLawNativeConfig):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-calibration-config'
    VERSION: ClassVar[str] = '1.0.0'

    def __post_init__(self) -> None:
        if (
            self.stage != "calibration"
            or self.retained_source is not None
            or self.retained_predecessors
            or self.native_owner.object_schema != 'empirical-lawhood/runtime/capability-manifest'
        ):
            raise ValueError("Finite response-law fresh calibration cannot import, replace or expand roots")

    @property
    def roots(self) -> tuple[FiniteResponseLawCalibrationRoot, ...]:
        return tuple(
            FiniteResponseLawCalibrationRoot(
                self.stage,
                i,
                sha256(str(seed_for("prefix", f"calibration.r{i:03d}")).encode()).hexdigest(),
                None,
            )
            for i in range(32)
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationInvocation(FiniteResponseLawNativeInvocation):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-calibration-invocation'
    VERSION: ClassVar[str] = '1.0.0'
    root: FiniteResponseLawCalibrationRoot

    def __post_init__(self) -> None:
        if (
            type(self.root) is not FiniteResponseLawCalibrationRoot
            or self.source.object_schema != FiniteResponseLawCalibrationConfig.SCHEMA
        ):
            raise ValueError("Finite response-law fresh invocation requires its calibration root and source")
        if self.phase == "prefix":
            valid = self.parent is None and self.word is None and self.purpose == "prefix"
        elif self.phase == "parent":
            valid = (
                self.parent == self.root.assigned_parent
                and self.word is None
                and self.purpose == "parent"
            )
        elif self.phase == "future":
            valid = (
                self.parent == self.root.assigned_parent
                and self.word is not None
                and self.word.direction_index in (0, 1)
                and self.purpose in ("future-1", "future-2")
            )
        else:
            valid = False
        if not valid:
            raise ValueError("Finite response-law fresh invocation changes its assigned native phase or action")

    @property
    def task_id(self) -> str:
        stem = f"{PROGRAMME}.{self.root.stage_unit}"
        if self.phase == "prefix":
            return f"{stem}.prefix.native"
        if self.phase == "parent":
            return f"{stem}.{self.parent}.parent.native"
        assert self.word is not None
        return f"{stem}.{self.parent}.{self.purpose}.{self.word.word_id}.native"

    @property
    def predecessor_segment_id(self) -> str | None:
        stem = f"{PROGRAMME}.{self.root.stage_unit}"
        if self.phase == "prefix":
            return None
        return (
            f"{stem}.prefix.native"
            if self.phase == "parent"
            else f"{stem}.{self.parent}.parent.native"
        )


def calibration_invocations(
    config: FiniteResponseLawCalibrationConfig,
) -> tuple[FiniteResponseLawCalibrationInvocation, ...]:
    identity = ObjectIdentity.from_record(config.spec_id, config)
    result = []
    for root in config.roots:
        result.append(FiniteResponseLawCalibrationInvocation(identity, root, "prefix", None, None, "prefix"))
        result.append(
            FiniteResponseLawCalibrationInvocation(
                identity, root, "parent", root.assigned_parent, None, "parent"
            )
        )
        for purpose in ("future-1", "future-2"):
            for word in config.words:
                result.append(
                    FiniteResponseLawCalibrationInvocation(
                        identity, root, "future", root.assigned_parent, word, purpose
                    )
                )
    if (len(result), sum(t.maximum_native_updates for t in result)) != (640, 751104):
        raise ValueError(
            "Finite response-law calibration differs from its full frozen root/menu/future/view budget"
        )
    return tuple(sorted(result, key=lambda t: t.task_id))
