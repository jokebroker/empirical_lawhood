"""Assigned Tier 1 native identities, separate from the exposed fixed rosters."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.methods.finite_response_law.science import PROGRAMME, seed_for, validate_assigned_seeds, validate_root_seeds
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import validate_sha256

from .evaluation_contracts import FiniteResponseLawEvaluationConfig, FiniteResponseLawEvaluationInvocation, FiniteResponseLawEvaluationRoot
from .fresh_contracts import FiniteResponseLawCalibrationConfig, FiniteResponseLawCalibrationInvocation, FiniteResponseLawCalibrationRoot
from .randomness import _assigned_stage_unit


def _seed(unit: str, scientific_seeds: tuple[tuple[str, int], ...]) -> str:
    return sha256(str(seed_for("prefix", unit, committed_seed=dict(scientific_seeds).get("prefix"))).encode()).hexdigest()


def _root_seeds(rows: tuple[tuple[str, int, int], ...], index: int) -> tuple[tuple[str, int], ...]:
    return tuple((p, seed) for p, i, seed in rows if i in (index, -1))


def _valid_config(
    config: FiniteResponseLawCalibrationConfig, namespace: str, stage: str, count: int
) -> bool:
    return (
        config.stage == stage
        and _assigned_stage_unit(f"{namespace}.r000", stage, count)
        and config.retained_source is None
        and not config.retained_predecessors
        and config.native_owner.object_schema
        == 'empirical-lawhood/runtime/capability-manifest'
    )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedCalibrationRoot(FiniteResponseLawCalibrationRoot):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-assigned-calibration-root'
    VERSION: ClassVar[str] = '1.0.0'
    cohort_namespace: str
    scientific_seeds: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        validate_root_seeds(self.cohort, self.scientific_seeds)
        if (
            self.cohort != "calibration"
            or type(self.index) is not int
            or not 0 <= self.index < 32
            or self.retained_root is not None
            or not _assigned_stage_unit(self.stage_unit, "calibration", 32)
            or self.seed_sha256 != _seed(self.stage_unit, self.scientific_seeds)
        ):
            raise ValueError(
                "Finite response-law assigned calibration root changes its stage, seed or unit"
            )

    @property
    def stage_unit(self) -> str:
        return f"{self.cohort_namespace}.r{self.index:03d}"

    @property
    def root_id(self) -> str:
        return f"{self.stage_unit}.prepared"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedCalibrationConfig(FiniteResponseLawCalibrationConfig):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-assigned-calibration-config'
    )
    VERSION: ClassVar[str] = '1.0.0'
    cohort_namespace: str
    assignment_sha256: str
    scientific_seeds: tuple[tuple[str, int, int], ...]

    def __post_init__(self) -> None:
        validate_assigned_seeds(self.stage, self.scientific_seeds)
        validate_sha256(self.assignment_sha256, field_name="assignment_sha256")
        if not _valid_config(self, self.cohort_namespace, "calibration", 32):
            raise ValueError(
                "Finite response-law assigned calibration config changes its source or cohort"
            )

    @property
    def spec_id(self) -> str:
        tag = sha256(self.cohort_namespace.encode()).hexdigest()[:16]
        return f"{PROGRAMME}.calibration.assigned-{tag}.native-config"

    @property
    def roots(self) -> tuple[FiniteResponseLawAssignedCalibrationRoot, ...]:
        return tuple(
            FiniteResponseLawAssignedCalibrationRoot(
                "calibration",
                i,
                _seed(f"{self.cohort_namespace}.r{i:03d}", _root_seeds(self.scientific_seeds, i)),
                None,
                self.cohort_namespace,
                _root_seeds(self.scientific_seeds, i),
            )
            for i in range(32)
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedCalibrationInvocation(FiniteResponseLawCalibrationInvocation):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-assigned-calibration-invocation'
    )
    VERSION: ClassVar[str] = '1.0.0'
    root: FiniteResponseLawAssignedCalibrationRoot

    def __post_init__(self) -> None:
        if (
            type(self.root) is not FiniteResponseLawAssignedCalibrationRoot
            or self.source.object_schema != FiniteResponseLawAssignedCalibrationConfig.SCHEMA
            or not _valid_invocation(self)
        ):
            raise ValueError(
                "Finite response-law assigned calibration invocation changes its root or phase"
            )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedEvaluationRoot(FiniteResponseLawEvaluationRoot):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-assigned-evaluation-root'
    VERSION: ClassVar[str] = '1.0.0'
    cohort_namespace: str
    scientific_seeds: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        validate_root_seeds(self.cohort, self.scientific_seeds)
        if (
            self.cohort != "prospective-evaluation"
            or type(self.index) is not int
            or not 0 <= self.index < 64
            or self.retained_root is not None
            or not _assigned_stage_unit(self.stage_unit, "prospective-evaluation", 64)
            or self.seed_sha256 != _seed(self.stage_unit, self.scientific_seeds)
        ):
            raise ValueError(
                "Finite response-law assigned evaluation root changes its stage, seed or unit"
            )

    @property
    def stage_unit(self) -> str:
        return f"{self.cohort_namespace}.r{self.index:03d}"

    @property
    def root_id(self) -> str:
        return f"{self.stage_unit}.prepared"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedEvaluationConfig(FiniteResponseLawEvaluationConfig):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-assigned-evaluation-config'
    )
    VERSION: ClassVar[str] = '1.0.0'
    cohort_namespace: str
    assignment_sha256: str
    scientific_seeds: tuple[tuple[str, int, int], ...]

    def __post_init__(self) -> None:
        validate_assigned_seeds(self.stage, self.scientific_seeds)
        validate_sha256(self.assignment_sha256, field_name="assignment_sha256")
        if not _valid_config(self, self.cohort_namespace, "prospective-evaluation", 64):
            raise ValueError(
                "Finite response-law assigned evaluation config changes its source or cohort"
            )

    @property
    def spec_id(self) -> str:
        tag = sha256(self.cohort_namespace.encode()).hexdigest()[:16]
        return f"{PROGRAMME}.prospective-evaluation.assigned-{tag}.native-config"

    @property
    def roots(self) -> tuple[FiniteResponseLawAssignedEvaluationRoot, ...]:
        return tuple(
            FiniteResponseLawAssignedEvaluationRoot(
                "prospective-evaluation",
                i,
                _seed(f"{self.cohort_namespace}.r{i:03d}", _root_seeds(self.scientific_seeds, i)),
                None,
                self.cohort_namespace,
                _root_seeds(self.scientific_seeds, i),
            )
            for i in range(64)
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedEvaluationInvocation(FiniteResponseLawEvaluationInvocation):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-assigned-evaluation-invocation'
    )
    VERSION: ClassVar[str] = '1.0.0'
    root: FiniteResponseLawAssignedEvaluationRoot

    def __post_init__(self) -> None:
        if (
            type(self.root) is not FiniteResponseLawAssignedEvaluationRoot
            or self.source.object_schema != FiniteResponseLawAssignedEvaluationConfig.SCHEMA
            or not _valid_invocation(self)
        ):
            raise ValueError(
                "Finite response-law assigned evaluation invocation changes its root or phase"
            )


def _valid_invocation(task: FiniteResponseLawCalibrationInvocation) -> bool:
    if task.phase == "prefix":
        return task.parent is None and task.word is None and task.purpose == "prefix"
    if task.phase == "parent":
        return (
            task.parent == task.root.assigned_parent
            and task.word is None
            and task.purpose == "parent"
        )
    return bool(
        task.phase == "future"
        and task.parent == task.root.assigned_parent
        and task.word is not None
        and task.word.direction_index in (0, 1)
        and task.purpose in ("future-1", "future-2")
    )


def assigned_calibration_invocations(
    config: FiniteResponseLawAssignedCalibrationConfig,
) -> tuple[FiniteResponseLawAssignedCalibrationInvocation, ...]:
    identity = ObjectIdentity.from_record(config.spec_id, config)
    tasks = [
        FiniteResponseLawAssignedCalibrationInvocation(identity, root, phase, parent, word, purpose)
        for root in config.roots
        for phase, parent, word, purpose in _native_menu(root, config.words)
    ]
    if (len(tasks), sum(t.maximum_native_updates for t in tasks)) != (640, 751104):
        raise ValueError("Finite response-law assigned calibration changes its frozen native budget")
    return tuple(sorted(tasks, key=lambda task: task.task_id))


def assigned_evaluation_invocations(
    config: FiniteResponseLawAssignedEvaluationConfig,
) -> tuple[FiniteResponseLawAssignedEvaluationInvocation, ...]:
    identity = ObjectIdentity.from_record(config.spec_id, config)
    tasks = [
        FiniteResponseLawAssignedEvaluationInvocation(identity, root, phase, parent, word, purpose)
        for root in config.roots
        for phase, parent, word, purpose in _native_menu(root, config.words)
    ]
    if (len(tasks), sum(t.maximum_native_updates for t in tasks)) != (1280, 1502208):
        raise ValueError("Finite response-law assigned evaluation changes its frozen native budget")
    return tuple(sorted(tasks, key=lambda task: task.task_id))


def _native_menu(root, words):
    yield "prefix", None, None, "prefix"
    yield "parent", root.assigned_parent, None, "parent"
    for purpose in ("future-1", "future-2"):
        for word in words:
            yield "future", root.assigned_parent, word, purpose


__all__ = [
    'FiniteResponseLawAssignedCalibrationConfig',
    'FiniteResponseLawAssignedCalibrationInvocation',
    'FiniteResponseLawAssignedCalibrationRoot',
    'FiniteResponseLawAssignedEvaluationConfig',
    'FiniteResponseLawAssignedEvaluationInvocation',
    'FiniteResponseLawAssignedEvaluationRoot',
    "assigned_calibration_invocations",
    "assigned_evaluation_invocations",
]
