"""Two prepared decisions and a separately declared conditional episode receiver.

This is a bounded join of existing prospective owners. It introduces neither a
trajectory controller nor another law/admission/evaluation engine.
"""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.time import ClockCoordinate
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)
from .controller_study import ImplementationBinding, ImplementationRole
from .nested_controller_evaluation import CommonStartControllerEvaluationPlan, CoupledRealizationControllerEvaluationPlan, PreparedInterfaceReducerRegistration, PreparedNativeRealizationCoupling


@dataclass(frozen=True, slots=True)
class SequenceStageContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/sequence-stage-contract'
    ordinal: int
    native_offset: D
    evaluation_plan: ObjectIdentity
    frozen_consumer: ObjectIdentity

    def __post_init__(self) -> None:
        validate_decimal(self.native_offset, field_name="native_offset", minimum=D(0))
        if (
            type(self.ordinal) is not int
            or self.ordinal not in (0, 1)
            or (self.ordinal == 0) != (self.native_offset == 0)
            or self.evaluation_plan.object_schema
            not in (
                CommonStartControllerEvaluationPlan.SCHEMA,
                CoupledRealizationControllerEvaluationPlan.SCHEMA,
            )
            or self.frozen_consumer.object_schema != 'empirical-lawhood/planning/frozen-feedback-consumer'
        ):
            raise ValueError("sequence stage requires its exact prepared owner and native offset")


@dataclass(frozen=True, slots=True)
class SequenceReceiverLimit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/sequence-receiver-limit'
    receiver_id: str
    unit: str
    observation_operator: ObjectIdentity
    lower: D | None
    upper: D | None
    numerical_tolerance: D

    def __post_init__(self) -> None:
        validate_stable_id(self.receiver_id, field_name="receiver_id")
        if not self.unit or (self.lower is None and self.upper is None):
            raise ValueError("sequence receiver requires native units and at least one bound")
        for value in (self.lower, self.upper):
            if value is not None:
                validate_decimal(value, field_name="receiver_bound")
        validate_decimal(self.numerical_tolerance, field_name="numerical_tolerance", minimum=D(0))
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("sequence receiver interval is reversed")


@dataclass(frozen=True, slots=True)
class BoundedSequenceEvaluationPlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/bounded-sequence-evaluation-plan'
    evaluation_plan_id: str
    roots: tuple[str, ...]
    policy_id: str
    request_id: str
    stages: tuple[SequenceStageContract, SequenceStageContract]
    native_clock: ClockCoordinate
    episode_duration: D
    source: ObjectIdentity
    qualified_joint_relation: ObjectIdentity
    numerical_views: tuple[str, str]
    receivers: tuple[SequenceReceiverLimit, ...]
    evaluator: ImplementationBinding
    reducer: PreparedInterfaceReducerRegistration
    realizations: tuple[PreparedNativeRealizationCoupling, ...]

    def __post_init__(self) -> None:
        for name in ("evaluation_plan_id", "policy_id", "request_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.roots, field_name="roots", allow_empty=False)
        require_sorted_unique_strings(
            self.numerical_views, field_name="numerical_views", allow_empty=False
        )
        if (
            len(self.stages) != 2
            or tuple(s.ordinal for s in self.stages) != (0, 1)
            or self.native_clock.coordinate != 0
            or len(self.numerical_views) != 2
            or self.evaluator.role is not ImplementationRole.OUTCOME_EVALUATOR
            or not self.receivers
            or len(self.receivers) > 32
            or tuple(r.receiver_id for r in self.receivers)
            != tuple(sorted({r.receiver_id for r in self.receivers}))
            or tuple(r.root_id for r in self.realizations) != self.roots
        ):
            raise ValueError("sequence plan changes its bounded census, receivers or owner")
        validate_decimal(self.episode_duration, field_name="episode_duration", minimum=D(0))
        if self.episode_duration <= self.stages[1].native_offset:
            raise ValueError("sequence second decision lacks a subsequent episode interval")
        if (
            self.reducer.capability_key != self.evaluator.reference.capability_key
            or self.reducer.capability_version != self.evaluator.reference.capability_version
            or self.reducer.config_sha256 != self.evaluator.config_sha256
            or self.reducer.implementation_sha256 != self.evaluator.implementation_sha256
        ):
            raise ValueError("sequence join substitutes its registered prepared controller-use owner")
