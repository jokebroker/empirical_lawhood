"""The separate, frozen 64-root prospective census; no source effects here."""

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.methods.finite_response_law.science import PROGRAMME, seed_for
from .fresh_contracts import FiniteResponseLawCalibrationConfig, FiniteResponseLawCalibrationInvocation, FiniteResponseLawCalibrationRoot


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationRoot(FiniteResponseLawCalibrationRoot):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-evaluation-root'
    VERSION: ClassVar[str] = "1.0.0"

    def __post_init__(self) -> None:
        if (
            self.cohort != "prospective-evaluation"
            or type(self.index) is not int
            or not 0 <= self.index < 64
            or self.retained_root is not None
            or self.seed_sha256
            != sha256(str(seed_for("prefix", self.stage_unit)).encode()).hexdigest()
        ):
            raise ValueError("Finite response-law evaluation requires its exact fresh independent root")

    @property
    def stage_unit(self) -> str:
        return f"prospective-evaluation.r{self.index:03d}"

    @property
    def root_id(self) -> str:
        return f"{PROGRAMME}.prospective-evaluation.prepared.r{self.index:03d}"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationConfig(FiniteResponseLawCalibrationConfig):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-evaluation-config'
    VERSION: ClassVar[str] = "1.0.0"

    def __post_init__(self) -> None:
        if (
            self.stage != "prospective-evaluation"
            or self.retained_source is not None
            or self.retained_predecessors
            or self.native_owner.object_schema != 'empirical-lawhood/runtime/capability-manifest'
        ):
            raise ValueError("Finite response-law evaluation cannot import, replace or expand roots")

    @property
    def roots(self) -> tuple[FiniteResponseLawEvaluationRoot, ...]:
        return tuple(
            FiniteResponseLawEvaluationRoot(
                "prospective-evaluation",
                i,
                sha256(str(seed_for("prefix", f"prospective-evaluation.r{i:03d}")).encode()).hexdigest(),
                None,
            )
            for i in range(64)
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationInvocation(FiniteResponseLawCalibrationInvocation):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-evaluation-invocation'
    VERSION: ClassVar[str] = "1.0.0"
    root: FiniteResponseLawEvaluationRoot

    def __post_init__(self) -> None:
        if (
            type(self.root) is not FiniteResponseLawEvaluationRoot
            or self.source.object_schema != FiniteResponseLawEvaluationConfig.SCHEMA
        ):
            raise ValueError("Finite response-law evaluation invocation requires its exact root and source")
        valid = (
            self.phase == "prefix"
            and self.parent is None
            and self.word is None
            and self.purpose == "prefix"
            or self.phase == "parent"
            and self.parent == self.root.assigned_parent
            and self.word is None
            and self.purpose == "parent"
            or self.phase == "future"
            and self.parent == self.root.assigned_parent
            and self.word is not None
            and self.word.direction_index in (0, 1)
            and self.purpose in ("future-1", "future-2")
        )
        if not valid:
            raise ValueError("Finite response-law evaluation changes its preassigned parent or finite menu")


def evaluation_invocations(config: FiniteResponseLawEvaluationConfig) -> tuple[FiniteResponseLawEvaluationInvocation, ...]:
    source = ObjectIdentity.from_record(config.spec_id, config)
    result = []
    for root in config.roots:
        result.append(FiniteResponseLawEvaluationInvocation(source, root, "prefix", None, None, "prefix"))
        result.append(
            FiniteResponseLawEvaluationInvocation(source, root, "parent", root.assigned_parent, None, "parent")
        )
        result.extend(
            FiniteResponseLawEvaluationInvocation(source, root, "future", root.assigned_parent, word, purpose)
            for purpose in ("future-1", "future-2")
            for word in config.words
        )
    if (len(result), sum(t.maximum_native_updates for t in result)) != (1280, 1502208):
        raise ValueError("Finite response-law evaluation changes its complete root/menu/future/view budget")
    return tuple(sorted(result, key=lambda t: t.task_id))
