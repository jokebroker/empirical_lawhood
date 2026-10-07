"""Closed Tier 1 native census; no historical stage is used as a new label."""

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.methods.finite_response_law.science import PROGRAMME, FiniteResponseLawScienceSpec, seed_for
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS, PreparedForceWord, PreparedNativeSpec, PreparedRoot, prepared_words
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_sha256,
)
from empirical_lawhood.planning.source_qualification import SourceQualificationRetainedPredecessor

CANARY_PARENT = "y-positive-256"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawNativeRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-native-root'
    cohort: str
    index: int
    seed_sha256: str
    retained_root: PreparedRoot | None

    def __post_init__(self) -> None:
        validate_sha256(self.seed_sha256, field_name="seed_sha256")
        if type(self.index) is not int:
            raise ValueError("Finite response-law root index must be an integer")
        if self.cohort == "native-canary":
            expected = sha256(
                str(seed_for("prefix", "canary.r000")).encode()
            ).hexdigest()
            if (
                self.index != 0
                or self.retained_root is not None
                or self.seed_sha256 != expected
            ):
                raise ValueError(
                    "Finite response-law Tier 1 canary is its single excluded committed root"
                )
        elif self.cohort == "retained-prepared-response":
            root = self.retained_root
            if (
                root is None
                or root.stage != 'qualification'
                or root.context != "prepared"
                or root.index != self.index
                or not 0 <= self.index < 16
                or root.seed_sha256 != self.seed_sha256
            ):
                raise ValueError(
                    "Finite response-law retained root must preserve the actual prepared-response physical unit"
                )
        else:
            raise ValueError("Finite response-law native root belongs to an inactive stage")

    @property
    def root_id(self) -> str:
        return (
            f"{PROGRAMME}.canary.prepared.r000"
            if self.retained_root is None
            else self.retained_root.root_id
        )

    @property
    def physical_unit_id(self) -> str:
        return f"{self.root_id}.seed.{self.seed_sha256}"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawNativeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-native-config'
    stage: str
    science: FiniteResponseLawScienceSpec
    native_owner: ObjectIdentity
    retained_source: PreparedNativeSpec | None
    retained_predecessors: tuple[SourceQualificationRetainedPredecessor, ...]

    def __post_init__(self) -> None:
        if (
            self.native_owner.object_schema
            != 'empirical-lawhood/runtime/capability-manifest'
        ):
            raise ValueError("Finite response-law source requires its exact installed native owner")
        require_sorted_unique_ids(
            self.retained_predecessors,
            attribute="segment_id",
            field_name="retained_predecessors",
        )
        if self.stage == "native-canary":
            if self.retained_source is not None or self.retained_predecessors:
                raise ValueError(
                    "fresh excluded canary cannot import a retained scientific root"
                )
        elif self.stage == "supplemental-development":
            source = self.retained_source
            if source is None or source.stage != 'qualification':
                raise ValueError(
                    "D1 requires an explicit retained Q source specification"
                )
            expected = {
                f"{r.root_id}.{p}.parent.native": r
                for r in source.roots
                if r.context == "prepared"
                for p in PARENTS
            }
            if {p.segment_id for p in self.retained_predecessors} != set(expected):
                raise ValueError(
                    "D1 requires exactly the eighty retained prepared-response parent handoffs"
                )
            for prior in self.retained_predecessors:
                root = expected[prior.segment_id]
                if (
                    prior.physical_independent_unit_id != root.physical_unit_id
                    or prior.end != 4368
                    or prior.native_clock_id != f"{PROGRAMME}.reference-clock"
                    or prior.view_ids
                    != tuple(f"{root.root_id}.flh-project.r{r}" for r in (1, 2))
                ):
                    raise ValueError(
                        "D1 predecessor changes native unit, end clock or views"
                    )
        else:
            raise ValueError("Finite response-law source stage is not active in Tier 1")

    @property
    def spec_id(self) -> str:
        return f"{PROGRAMME}.{self.stage}.native-config"

    @property
    def roots(self) -> tuple[FiniteResponseLawNativeRoot, ...]:
        if self.retained_source is None:
            seed = sha256(str(seed_for("prefix", "canary.r000")).encode()).hexdigest()
            return (FiniteResponseLawNativeRoot("native-canary", 0, seed, None),)
        return tuple(
            FiniteResponseLawNativeRoot("retained-prepared-response", r.index, r.seed_sha256, r)
            for r in self.retained_source.roots
            if r.context == "prepared"
        )

    @property
    def words(self) -> tuple[PreparedForceWord, ...]:
        return tuple(w for w in prepared_words() if w.direction_index in (0, 1))


@dataclass(frozen=True, slots=True)
class FiniteResponseLawNativeInvocation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-native-invocation'
    source: ObjectIdentity
    root: FiniteResponseLawNativeRoot
    phase: str
    parent: str | None
    word: PreparedForceWord | None
    purpose: str

    def __post_init__(self) -> None:
        if self.source.object_schema != FiniteResponseLawNativeConfig.SCHEMA:
            raise ValueError("Finite response-law invocation requires the versioned native config")
        canary = self.root.cohort == "native-canary"
        if self.phase == "prefix":
            valid = (
                canary
                and self.parent is None
                and self.word is None
                and self.purpose == "prefix"
            )
        elif self.phase == "parent":
            valid = (
                canary
                and self.parent == CANARY_PARENT
                and self.word is None
                and self.purpose == "parent"
            )
        elif self.phase == "future":
            valid = (
                self.parent in ((CANARY_PARENT,) if canary else PARENTS)
                and self.word is not None
                and self.word.direction_index in (0, 1)
                and self.purpose
                in (("future-1", "future-2") if canary else ("future-2",))
            )
        else:
            valid = False
        if not valid:
            raise ValueError(
                "Finite response-law invocation changes its native phase, chart, parent or future purpose"
            )

    @property
    def task_id(self) -> str:
        root = (
            f"{PROGRAMME}.native-canary.r000"
            if self.root.cohort == "native-canary"
            else f"{PROGRAMME}.supplemental-development.r{self.root.index:03d}"
        )
        if self.phase == "prefix":
            return f"{root}.prefix.native"
        if self.phase == "parent":
            return f"{root}.{self.parent}.parent.native"
        assert self.word is not None
        return f"{root}.{self.parent}.{self.purpose}.{self.word.word_id}.native"

    @property
    def clocks(self) -> tuple[int, int]:
        return {"prefix": (0, 4096), "parent": (4096, 4368), "future": (4368, 4560)}[
            self.phase
        ]

    @property
    def maximum_native_updates(self) -> int:
        start, end = self.clocks
        return 3 * (end - start)

    @property
    def predecessor_segment_id(self) -> str | None:
        if self.phase == "prefix":
            return None
        if self.root.cohort == "retained-prepared-response":
            return f"{self.root.root_id}.{self.parent}.parent.native"
        stem = f"{PROGRAMME}.native-canary.r000"
        return (
            f"{stem}.prefix.native"
            if self.phase == "parent"
            else f"{stem}.{self.parent}.parent.native"
        )

    @property
    def dependency_task_ids(self) -> tuple[str, ...]:
        predecessor = self.predecessor_segment_id
        return (
            ()
            if predecessor is None or self.root.cohort == "retained-prepared-response"
            else (predecessor,)
        )


def native_invocations(config: FiniteResponseLawNativeConfig) -> tuple[FiniteResponseLawNativeInvocation, ...]:
    from .assigned_contracts import FiniteResponseLawAssignedCalibrationConfig, FiniteResponseLawAssignedEvaluationConfig, assigned_calibration_invocations, assigned_evaluation_invocations
    from .evaluation_contracts import FiniteResponseLawEvaluationConfig, evaluation_invocations
    from .fresh_contracts import FiniteResponseLawCalibrationConfig, calibration_invocations

    if type(config) is FiniteResponseLawAssignedEvaluationConfig:
        return assigned_evaluation_invocations(config)
    if type(config) is FiniteResponseLawAssignedCalibrationConfig:
        return assigned_calibration_invocations(config)
    if type(config) is FiniteResponseLawEvaluationConfig:
        return evaluation_invocations(config)
    if type(config) is FiniteResponseLawCalibrationConfig:
        return calibration_invocations(config)
    source = ObjectIdentity.from_record(config.spec_id, config)
    tasks = []
    for root in config.roots:
        canary = root.cohort == "native-canary"
        if canary:
            tasks.append(
                FiniteResponseLawNativeInvocation(source, root, "prefix", None, None, "prefix")
            )
            tasks.append(
                FiniteResponseLawNativeInvocation(
                    source, root, "parent", CANARY_PARENT, None, "parent"
                )
            )
        for parent in (CANARY_PARENT,) if canary else PARENTS:
            for purpose in ("future-1", "future-2") if canary else ("future-2",):
                for word in config.words:
                    tasks.append(
                        FiniteResponseLawNativeInvocation(
                            source, root, "future", parent, word, purpose
                        )
                    )
    expected = (20, 23472) if config.stage == "native-canary" else (720, 414720)
    if (len(tasks), sum(t.maximum_native_updates for t in tasks)) != expected:
        raise ValueError("Finite response-law native roster differs from the frozen acquisition budget")
    return tuple(sorted(tasks, key=lambda t: t.task_id))
