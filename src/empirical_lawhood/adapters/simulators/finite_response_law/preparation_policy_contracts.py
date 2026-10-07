"Additive preparation-policy +400 preparation records.\n\nThese records keep the existing native-source and predecessor-bound source-qualification decoders unchanged.  They bind\nthe nine preparation schedules and the retained-prefix census used by preparation-policy development;\nexecution authority remains external.\n"

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, require_sorted_unique_ids
from empirical_lawhood.planning.source_qualification import SourceQualificationRetainedPredecessor
from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec, PROGRAMME
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedForceWord, PreparedNativeSpec, prepared_words


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationSchedule(CanonicalRecord):
    """One of the exact nine +400 preparation policies."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-preparation-schedule'
    VERSION: ClassVar[str] = '1.0.0'
    schedule_id: str
    duration_ticks: int
    recovery_ticks: int
    sign: int
    excursion: Decimal = Decimal("0.125")
    coupling: str = "Y"
    handoff_ticks: int = 400

    def __post_init__(self) -> None:
        hold = self.schedule_id == "hold"
        if hold:
            valid = self.duration_ticks == 0 and self.recovery_ticks == 400 and self.sign == 0
        else:
            expected = (
                f"y-{'negative' if self.sign == -1 else 'positive'}-"
                f"{self.duration_ticks}-recovery-{self.recovery_ticks:03d}"
            )
            valid = (
                self.duration_ticks in (128, 256)
                and self.recovery_ticks in (16, 144)
                and self.sign in (-1, 1)
                and self.schedule_id == expected
                and self.duration_ticks + self.recovery_ticks <= self.handoff_ticks
            )
        if (
            not valid
            or self.excursion != Decimal("0.125")
            or self.coupling != "Y"
            or self.handoff_ticks != 400
        ):
            raise ValueError("preparation-policy schedule differs from the frozen nine-policy +400 chart")

    @property
    def pulse_start_tick(self) -> int:
        return self.handoff_ticks - self.duration_ticks - self.recovery_ticks


def preparation_policy_schedules() -> tuple[FiniteResponseLawPreparationSchedule, ...]:
    result = [FiniteResponseLawPreparationSchedule("hold", 0, 400, 0)]
    for duration in (128, 256):
        for recovery in (16, 144):
            for sign in (-1, 1):
                result.append(
                    FiniteResponseLawPreparationSchedule(
                        f"y-{'negative' if sign == -1 else 'positive'}-"
                        f"{duration}-recovery-{recovery:03d}",
                        duration,
                        recovery,
                        sign,
                    )
                )
    hold, *active = result
    return (hold, *sorted(active, key=lambda value: value.schedule_id))


PREPARATION_POLICY_SCHEDULES = preparation_policy_schedules()
PREPARATION_POLICY_SCHEDULE_IDS = tuple(schedule.schedule_id for schedule in PREPARATION_POLICY_SCHEDULES)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyRetainedPrefix(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-preparation-policy-retained-prefix'
    declaration: SourceQualificationRetainedPredecessor
    source_spec: ObjectIdentity
    native_result: ObjectIdentity
    prefix_features: tuple[tuple[Decimal, ...], ...]
    feature_instrument: ObjectIdentity

    def __post_init__(self) -> None:
        if (
            self.source_spec.object_schema != PreparedNativeSpec.SCHEMA
            or self.native_result.object_schema
            != 'empirical-lawhood/simulators/prepared-response/prepared-native-task-result'
            or self.native_result.object_id != f"{self.declaration.segment_id}.result"
            or len(self.prefix_features) != 2
            or any(
                len(view) != 24
                or any(not isinstance(value, Decimal) or not value.is_finite() for value in view)
                for view in self.prefix_features
            )
            or self.feature_instrument.object_schema
            != 'empirical-lawhood/simulators/finite-response-law/finite-response-law-reference-instrument'
        ):
            raise ValueError("preparation-policy retained prefix loses its exact source/result identity")


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-preparation-policy-root'
    cohort: str
    index: int
    retained_prefix: FiniteResponseLawPreparationPolicyRetainedPrefix | None

    def __post_init__(self) -> None:
        if type(self.index) is not int:
            raise ValueError("preparation-policy root index must be an integer")
        if self.cohort in ("prepared-response", "information-response-prediction"):
            limit = 8 if self.cohort == "prepared-response" else 16
            predecessor = self.retained_prefix
            expected_root = f"prepared-response.{'qualification' if self.cohort == 'prepared-response' else 'prospective-evaluation'}.prepared.r{self.index:03d}"
            valid = (
                0 <= self.index < limit
                and predecessor is not None
                and predecessor.declaration.segment_id == f"{expected_root}.prefix.native"
                and predecessor.declaration.native_clock_id == f"{PROGRAMME}.reference-clock"
                and predecessor.declaration.end == Decimal(4096)
                and predecessor.declaration.view_ids == (f"{self.stage_unit}.project",)
            )
        else:
            valid = False
        if not valid:
            raise ValueError("preparation-policy root differs from the exact 8-prepared-response/16-information-response-prediction census")

    @property
    def stage_unit(self) -> str:
        position = self.index if self.cohort == "prepared-response" else 8 + self.index
        return f"preparation-screening.r{position:03d}"

    @property
    def root_id(self) -> str:
        return (
            f"prepared-response.{'qualification' if self.cohort == 'prepared-response' else 'prospective-evaluation'}.prepared.r{self.index:03d}"
        )

    @property
    def physical_unit_id(self) -> str:
        assert self.retained_prefix is not None
        return self.retained_prefix.declaration.physical_independent_unit_id


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyNativeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-preparation-policy-native-config'
    stage: str
    science: FiniteResponseLawScienceSpec
    native_owner: ObjectIdentity
    retained_prefixes: tuple[FiniteResponseLawPreparationPolicyRetainedPrefix, ...]

    def __post_init__(self) -> None:
        if self.native_owner.object_schema != 'empirical-lawhood/runtime/capability-manifest':
            raise ValueError("preparation-policy source requires its exact installed native owner")
        require_sorted_unique_ids(
            tuple(row.declaration for row in self.retained_prefixes),
            attribute="segment_id",
            field_name="retained_prefixes",
        )
        if self.stage == "preparation-screening":
            expected = {
                *(f"prepared-response.qualification.prepared.r{i:03d}.prefix.native" for i in range(8)),
                *(f"prepared-response.prospective-evaluation.prepared.r{i:03d}.prefix.native" for i in range(16)),
            }
            valid = {row.declaration.segment_id for row in self.retained_prefixes} == expected
        else:
            valid = False
        if not valid:
            raise ValueError("preparation-policy config changes its exact 24-prefix census")

    @property
    def spec_id(self) -> str:
        return f"{PROGRAMME}.{self.stage}.preparation-policy-native-config"

    @property
    def roots(self) -> tuple[FiniteResponseLawPreparationPolicyRoot, ...]:
        by_id = {row.declaration.segment_id: row for row in self.retained_prefixes}
        return tuple(
            FiniteResponseLawPreparationPolicyRoot(
                cohort,
                index,
                by_id[
                    f"prepared-response.{'qualification' if cohort == 'prepared-response' else 'prospective-evaluation'}.prepared.r{index:03d}.prefix.native"
                ],
            )
            for cohort, count in (("prepared-response", 8), ("information-response-prediction", 16))
            for index in range(count)
        )

    @property
    def schedules(self) -> tuple[FiniteResponseLawPreparationSchedule, ...]:
        return PREPARATION_POLICY_SCHEDULES

    @property
    def words(self) -> tuple[PreparedForceWord, ...]:
        return tuple(word for word in prepared_words() if word.direction_index in (0, 1))


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyNativeInvocation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-preparation-policy-native-invocation'
    source: ObjectIdentity
    root: FiniteResponseLawPreparationPolicyRoot
    phase: str
    schedule: FiniteResponseLawPreparationSchedule | None
    word: PreparedForceWord | None
    purpose: str

    def __post_init__(self) -> None:
        if self.source.object_schema != FiniteResponseLawPreparationPolicyNativeConfig.SCHEMA:
            raise ValueError("preparation-policy invocation requires its versioned source config")
        if self.phase == "preparation":
            valid = self.schedule in PREPARATION_POLICY_SCHEDULES and self.word is None and self.purpose == "parent"
        elif self.phase == "future":
            valid = (
                self.schedule in PREPARATION_POLICY_SCHEDULES
                and self.word is not None
                and self.word.direction_index in (0, 1)
                and self.purpose in ("future-1", "future-2")
            )
        else:
            valid = False
        if not valid:
            raise ValueError("preparation-policy invocation changes phase, schedule, chart or purpose")

    @property
    def task_id(self) -> str:
        stem = f"{PROGRAMME}.{self.root.stage_unit}"
        assert self.schedule is not None
        if self.phase == "preparation":
            return f"{stem}.{self.schedule.schedule_id}.preparation.native"
        assert self.word is not None
        return (
            f"{stem}.{self.schedule.schedule_id}.{self.purpose}."
            f"{self.word.word_id}.native"
        )

    @property
    def clocks(self) -> tuple[int, int]:
        return {
            "preparation": (4096, 4496),
            "future": (4496, 4688),
        }[self.phase]

    @property
    def maximum_native_updates(self) -> int:
        start, end = self.clocks
        return 3 * (end - start)

    @property
    def predecessor_segment_id(self) -> str | None:
        stem = f"{PROGRAMME}.{self.root.stage_unit}"
        if self.phase == "preparation":
            assert self.root.retained_prefix is not None
            return self.root.retained_prefix.declaration.segment_id
        assert self.schedule is not None
        return f"{stem}.{self.schedule.schedule_id}.preparation.native"

    @property
    def dependency_task_ids(self) -> tuple[str, ...]:
        predecessor = self.predecessor_segment_id
        if self.phase == "preparation":
            return ()
        assert predecessor is not None
        return (predecessor,)


def preparation_policy_native_invocations(
    config: FiniteResponseLawPreparationPolicyNativeConfig,
) -> tuple[FiniteResponseLawPreparationPolicyNativeInvocation, ...]:
    source = ObjectIdentity.from_record(config.spec_id, config)
    result: list[FiniteResponseLawPreparationPolicyNativeInvocation] = []
    for root in config.roots:
        for schedule in config.schedules:
            result.append(
                FiniteResponseLawPreparationPolicyNativeInvocation(
                    source, root, "preparation", schedule, None, "parent"
                )
            )
            for purpose in ("future-1", "future-2"):
                for word in config.words:
                    result.append(
                        FiniteResponseLawPreparationPolicyNativeInvocation(
                            source, root, "future", schedule, word, purpose
                        )
                    )
    expected = (4_104, 2_498_688)
    actual = (len(result), sum(row.maximum_native_updates for row in result))
    if actual != expected:
        raise ValueError("preparation-policy graph differs from the frozen task/update census")
    return tuple(sorted(result, key=lambda row: row.task_id))
