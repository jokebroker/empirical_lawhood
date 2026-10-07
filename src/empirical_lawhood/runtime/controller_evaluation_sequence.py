"""Bounded sequence operands/reduction invoked by NestedControllerUseEvaluator."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar, TYPE_CHECKING

from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)
from empirical_lawhood.planning.bounded_sequence_evaluation import BoundedSequenceEvaluationPlan
from .canonical_record_archive import CanonicalRecordArchive
from .controller_compiler import CompiledDeliveryControllerStudy
from .controller_runtime import DeliveryControllerTickReceipt, TickDisposition
from .controller_evaluation_trajectory import TrajectoryCallbackLink

if TYPE_CHECKING:
    from .controller_evaluation_nested import NestedControllerUseEvaluator


@dataclass(frozen=True, slots=True)
class SequenceChildEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/sequence-child-evidence'
    ordinal: int
    revealed: CanonicalRecordArchive
    compiled: CanonicalRecordArchive | None
    tick: CanonicalRecordArchive | None
    link: TrajectoryCallbackLink | None

    def __post_init__(self) -> None:
        if (
            type(self.ordinal) is not int
            or self.ordinal not in (0, 1)
            or self.revealed.subject.object_schema
            not in (
                'empirical-lawhood/runtime/revealed-prepared-policy-bundle',
                'empirical-lawhood/runtime/revealed-prepared-forecast-policy-bundle',
            )
            or len({self.compiled is None, self.tick is None, self.link is None}) != 1
            or (self.ordinal == 1 and self.tick is None)
        ):
            raise ValueError("sequence child loses its actual prepared evaluation or owner lineage")
        if self.compiled is not None and self.tick is not None:
            if (
                self.compiled.subject.object_schema != CompiledDeliveryControllerStudy.SCHEMA
                or self.tick.subject.object_schema != DeliveryControllerTickReceipt.SCHEMA
            ):
                raise ValueError("sequence child substitutes another compiled/tick schema")


@dataclass(frozen=True, slots=True)
class SequenceEpisodeMeasurement(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/sequence-episode-measurement'
    root: str
    source: ObjectIdentity
    realization: ObjectIdentity
    native_start: D
    elapsed: D
    numerical_views: tuple[str, str]
    native_artifact: ArtifactIdentity
    native_receipt: ObjectIdentity
    actual_measurements: tuple[ArtifactIdentity, ArtifactIdentity]
    references: tuple[ArtifactIdentity, ...]
    values: tuple[tuple[NamedDecimal, ...], tuple[NamedDecimal, ...]]
    observation_operators: tuple[tuple[str, ObjectIdentity], ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.root, field_name="root")
        validate_decimal(self.native_start, field_name="native_start", minimum=D(0))
        validate_decimal(self.elapsed, field_name="elapsed", minimum=D(0))
        require_sorted_unique_strings(
            self.numerical_views, field_name="numerical_views", allow_empty=False
        )
        if (
            len(self.numerical_views) != 2
            or len(self.actual_measurements) != 2
            or len(self.values) != 2
            or not self.references
            or any(len({v.value_id for v in row}) != len(row) for row in self.values)
            or len(set(self.actual_measurements)) != 2
            or tuple(k for k, _ in self.observation_operators)
            != tuple(sorted({v.value_id for v in self.values[0]}))
        ):
            raise ValueError(
                "sequence measurement omits a numerical view or actual/reference custody"
            )


@dataclass(frozen=True, slots=True)
class RevealedBoundedSequence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/revealed-bounded-sequence'
    bundle_id: str
    plan: ObjectIdentity
    root: str
    realization: ObjectIdentity
    children: tuple[SequenceChildEvidence, ...]
    episode: SequenceEpisodeMeasurement | None

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        validate_stable_id(self.root, field_name="root")
        if (
            self.plan.object_schema != BoundedSequenceEvaluationPlan.SCHEMA
            or not 1 <= len(self.children) <= 2
            or tuple(c.ordinal for c in self.children) != tuple(range(len(self.children)))
        ):
            raise ValueError("sequence requires its fixed plan and complete actual decision prefix")


@dataclass(frozen=True, slots=True)
class BoundedSequenceUnitEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/bounded-sequence-unit-evaluation'
    evaluation_id: str
    plan: ObjectIdentity
    bundle: ObjectIdentity
    root: str
    independent_unit_count: int
    admitted_stages: int
    evaluable: bool
    success: bool
    false_admission: bool | None
    joint_receivers: GateStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        validate_stable_id(self.root, field_name="root")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.independent_unit_count != 1
            or self.admitted_stages not in (0, 1, 2)
            or (self.evaluable and self.false_admission is None)
            or (
                self.success
                and (
                    not self.evaluable
                    or self.admitted_stages != 2
                    or self.false_admission is not False
                    or self.joint_receivers is not GateStatus.PASS
                )
            )
        ):
            raise ValueError("sequence event confuses unknown, refused or successful actual use")


def _read(archive: CanonicalRecordArchive, kind: type[CanonicalRecord]) -> CanonicalRecord:
    record = decode_canonical_bytes(archive.unpack(), kind, maximum_bytes=archive.decoded_bytes)
    if (
        archive.subject.object_schema != record.SCHEMA
        or archive.subject.object_fingerprint != record.fingerprint()
    ):
        raise ValueError("sequence owner archive substitutes its canonical record")
    return record


def evaluate_sequence(
    plan: BoundedSequenceEvaluationPlan,
    bundle: RevealedBoundedSequence,
    owner: NestedControllerUseEvaluator,
) -> BoundedSequenceUnitEvaluation:
    # Deferred import keeps the bounded reducer behind its sole owning service.
    from .controller_evaluation_nested import RevealedPreparedPolicyBundle, RevealedPreparedForecastPolicyBundle, PreparedPolicyUnitEvaluation, PreparedFutureDisposition, SealedPreparedForecastPolicyBundle

    identity = ObjectIdentity.from_record(plan.evaluation_plan_id, plan)
    if bundle.plan != identity or bundle.root not in plan.roots:
        raise ValueError("sequence substitutes its frozen plan or assigned independent root")
    realization = next(r for r in plan.realizations if r.root_id == bundle.root)
    if realization.scenario != bundle.realization:
        raise ValueError("sequence changes its assigned physical realization")
    realization_id = ObjectIdentity.from_record(realization.coupling_id, realization)
    units: list[PreparedPolicyUnitEvaluation] = []
    previous_tick = None
    start = None
    native_custody: set[tuple[ArtifactIdentity | None, ObjectIdentity | None]] = set()
    for child in bundle.children:
        kind = (
            RevealedPreparedForecastPolicyBundle
            if child.revealed.subject.object_schema == RevealedPreparedForecastPolicyBundle.SCHEMA
            else RevealedPreparedPolicyBundle
        )
        revealed = _read(child.revealed, kind)
        assert isinstance(revealed, RevealedPreparedPolicyBundle)
        compiled = (
            None if child.compiled is None else _read(child.compiled, CompiledDeliveryControllerStudy)
        )
        assert compiled is None or isinstance(compiled, CompiledDeliveryControllerStudy)
        unit = owner.evaluate_prepared_unit(
            revealed=revealed, compiled=compiled if revealed.sealed.instance is not None else None
        )
        if unit.revealed_bundle != child.revealed.subject:
            raise ValueError("sequence revealed child uses another exact owner identity")
        contract = plan.stages[child.ordinal]
        if (
            unit.root_id != bundle.root
            or unit.policy_id != plan.policy_id
            or unit.evaluation_plan != contract.evaluation_plan
        ):
            raise ValueError("sequence child changes its root, policy or prepared plan")
        if any(
            f.innovation_coupling != realization_id
            for f in revealed.sealed.design.evaluation_plan.futures
        ):
            raise ValueError("sequence child changes its frozen native realization coupling")
        if units and not units[-1].admitted:
            raise ValueError("sequence cannot continue after a refused preceding stage")
        if child.tick is None:
            if unit.admitted or child.ordinal != 0:
                raise ValueError("admitted sequence stage has no actual owner tick")
        else:
            assert child.compiled is not None and child.link is not None
            tick = _read(child.tick, DeliveryControllerTickReceipt)
            assert isinstance(tick, DeliveryControllerTickReceipt) and isinstance(
                compiled, CompiledDeliveryControllerStudy
            )
            observation = tick.commitment.observation
            if start is None:
                start = observation.coordinate.coordinate
            expected_time = start + contract.native_offset
            link = child.link
            if (
                tick.compiled_study != child.compiled.subject
                or child.compiled.subject
                != ObjectIdentity.from_record(compiled.compiled_study_id, compiled)
                or child.tick.subject != ObjectIdentity.from_record(tick.tick_id, tick)
                or (
                    revealed.sealed.forecast_parent.decision
                    if isinstance(revealed.sealed, SealedPreparedForecastPolicyBundle)
                    else revealed.sealed.commitment
                )
                != tick.commitment
                or compiled.study.prospective_evaluation != contract.evaluation_plan
                or compiled.study.instance_binding != tick.commitment.instance_binding
                or tick.commitment.instance_binding.frozen_recipe != contract.frozen_consumer
                or observation.independent_unit_id != bundle.root
                or observation.coordinate.coordinate != expected_time
                or tick.commitment.commitment_coordinate != observation.coordinate
                or observation.coordinate.clock_id != plan.native_clock.clock_id
                or observation.coordinate.time_unit != plan.native_clock.time_unit
                or observation.coordinate.coordinate_frame != plan.native_clock.coordinate_frame
                or observation.coordinate.origin is not plan.native_clock.origin
                or (link.root, link.index, link.previous_tick)
                != (bundle.root, child.ordinal, previous_tick)
                or link.causal_input not in observation.input_artifacts
                or not any(
                    a.artifact_id == link.link_id
                    and a.sha256 == link.fingerprint()
                    and a.payload_schema == link.SCHEMA
                    for a in observation.input_artifacts
                )
                or (unit.admitted and tick.disposition is TickDisposition.NONATTEMPT)
                or (not unit.admitted and tick.disposition is not TickDisposition.NONATTEMPT)
            ):
                raise ValueError(
                    "sequence substitutes a decision, native cutoff, consumer or predecessor"
                )
            for outcome in revealed.outcomes:
                if (
                    outcome.delivery_trace is not None
                    and outcome.delivery_trace.commitment
                    == ObjectIdentity.from_record(tick.commitment.commitment_id, tick.commitment)
                    and outcome.delivery_trace != tick.delivery_trace
                ):
                    raise ValueError(
                        "sequence measurement substitutes another actual delivery trace"
                    )
            previous_tick = child.tick.subject
        native_custody.update(
            (locator.artifact, locator.source_receipt)
            for locator in revealed.sealed.locators
            if locator.disposition is PreparedFutureDisposition.COMPLETED
        )
        units.append(unit)

    reasons: set[str] = set()
    unknown = False
    failure = False
    for unit in units:
        if not unit.admitted:
            if any(
                d
                in (
                    PreparedFutureDisposition.OPERATIONAL_STOP,
                    PreparedFutureDisposition.COMPLETION_UNKNOWN,
                    PreparedFutureDisposition.INVALID_OBSERVATION,
                )
                for d in unit.dispositions
            ):
                unknown = True
                reasons.add("UNKNOWN_STAGE_DISPOSITION")
            else:
                reasons.add("KNOWN_STAGE_NONATTEMPT")
            continue
        statuses = (unit.native_target, unit.delivery, unit.preservation, unit.numerics)
        unknown |= any(v is GateStatus.UNEVALUABLE for v in statuses)
        failure |= any(v is GateStatus.FAIL for v in statuses)
        if unknown:
            reasons.add("UNKNOWN_ADMITTED_STAGE_OUTCOME")
        if unit.admitted_failure:
            reasons.add("PREPARED_OWNER_OBLIGATION_NOT_PASSED")
    if len(units) == 1 and units[0].admitted:
        unknown = True
        reasons.add("MISSING_SECOND_OWNER_DECISION")
    joint = GateStatus.UNEVALUABLE
    both = len(units) == 2 and all(u.admitted for u in units)
    if not both and bundle.episode is not None:
        raise ValueError("refused sequence invents a completed conditional episode")
    if both:
        measured = bundle.episode
        if measured is None:
            unknown = True
            reasons.add("MISSING_CONDITIONAL_EPISODE_MEASUREMENTS")
        else:
            if (
                native_custody != {(measured.native_artifact, measured.native_receipt)}
                or measured.root != bundle.root
                or measured.realization != bundle.realization
                or measured.source != plan.source
                or measured.numerical_views != plan.numerical_views
                or measured.native_start != start
                or measured.elapsed != plan.episode_duration
                or measured.observation_operators
                != tuple((r.receiver_id, r.observation_operator) for r in plan.receivers)
            ):
                raise ValueError(
                    "sequence joint receiver substitutes its source, realization or horizon"
                )
            values = tuple({v.value_id: v for v in row} for row in measured.values)
            expected = {r.receiver_id for r in plan.receivers}
            if any(set(row) != expected for row in values):
                raise ValueError("sequence joint receiver census is incomplete or substituted")
            passed = True
            for limit in plan.receivers:
                pair = tuple(row[limit.receiver_id] for row in values)
                if any(v.unit != limit.unit for v in pair):
                    raise ValueError("sequence joint receiver changes its native unit")
                if abs(pair[0].value - pair[1].value) > limit.numerical_tolerance:
                    passed = False
                    reasons.add("JOINT_NUMERICAL_CORRESPONDENCE_FAILED")
                if any(
                    (limit.lower is not None and v.value < limit.lower)
                    or (limit.upper is not None and v.value > limit.upper)
                    for v in pair
                ):
                    passed = False
                    reasons.add("CONDITIONAL_JOINT_RECEIVER_FAILED")
            joint = GateStatus.PASS if passed else GateStatus.FAIL
            failure |= not passed
    admitted = sum(u.admitted for u in units)
    return BoundedSequenceUnitEvaluation(
        f"sequence-unit.{bundle.bundle_id}",
        identity,
        ObjectIdentity.from_record(bundle.bundle_id, bundle),
        bundle.root,
        1,
        admitted,
        not unknown,
        both and not unknown and all(u.task_success for u in units) and joint is GateStatus.PASS,
        True if failure else None if unknown else False,
        joint,
        tuple(sorted(reasons)),
    )
