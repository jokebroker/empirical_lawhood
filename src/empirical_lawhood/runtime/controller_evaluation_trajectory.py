"""Typed trajectory operands/reduction, invoked only by NestedControllerUseEvaluator.

Outcome access cannot select or deliver. Full owner records are read by exact
identity from custody, one callback at a time; callbacks never increase n.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar, Protocol
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal, ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.planning.trajectory_controller_evaluation import TrajectoryControllerEvaluationPlan
from empirical_lawhood.planning.finite_response_geometry import FiniteBinary64ResponseBound
from .controller_compiler import CompiledDeliveryControllerStudy
from .controller_runtime import DeliveryControllerTickReceipt, TickDisposition


@dataclass(frozen=True, slots=True)
class TrajectoryCallbackLink(CanonicalRecord):
    """Prospective link retained in the owner's exact runtime observation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/trajectory-callback-link'
    link_id: str
    root: str
    index: int
    previous_tick: ObjectIdentity | None
    causal_input: ArtifactIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.link_id, field_name="link_id")
        validate_stable_id(self.root, field_name="root")
        if (
            type(self.index) is not int
            or self.index < 0
            or (self.index == 0) != (self.previous_tick is None)
        ):
            raise ValueError("trajectory link loses its causal predecessor")
        if (
            self.previous_tick is not None
            and self.previous_tick.object_schema != DeliveryControllerTickReceipt.SCHEMA
        ):
            raise ValueError("trajectory predecessor must be an actual owner tick")


@dataclass(frozen=True, slots=True)
class TrajectoryCallbackOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/trajectory-callback-outcome'
    index: int
    compiled: ObjectIdentity
    tick: ObjectIdentity
    link: ObjectIdentity
    # View-major raw measured receivers; missingness is an empty tuple.
    labels: tuple[tuple[NamedDecimal, ...], ...]

    def __post_init__(self) -> None:
        if (
            type(self.index) is not int
            or self.index < 0
            or self.compiled.object_schema != CompiledDeliveryControllerStudy.SCHEMA
            or self.tick.object_schema != DeliveryControllerTickReceipt.SCHEMA
            or self.link.object_schema != TrajectoryCallbackLink.SCHEMA
        ):
            raise ValueError("trajectory callback owner identities differ")
        if any(len({v.value_id for v in view}) != len(view) for view in self.labels):
            raise ValueError("duplicate trajectory measurement receiver")


@dataclass(frozen=True, slots=True)
class RevealedTrajectoryBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/revealed-trajectory-bundle'
    bundle_id: str
    root: str
    plan: ObjectIdentity
    callbacks: tuple[TrajectoryCallbackOutcome, ...]
    native_complete: bool
    completion_seconds: Decimal | None
    reference_completion_seconds: Decimal | None
    raw_evidence: tuple[ObjectIdentity, ...]
    raw_path_valid: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        validate_stable_id(self.root, field_name="root")
        if self.plan.object_schema != TrajectoryControllerEvaluationPlan.SCHEMA or any(
            type(v) is not bool for v in (self.native_complete, self.raw_path_valid)
        ):
            raise ValueError("trajectory plan or completion/validity type differs")
        if tuple(c.index for c in self.callbacks) != tuple(range(len(self.callbacks))):
            raise ValueError("trajectory callbacks must be a complete prefix")
        for value in (self.completion_seconds, self.reference_completion_seconds):
            if value is not None and (not value.is_finite() or value < 0):
                raise ValueError("nonfinite/negative measured completion time")
        if not self.raw_evidence:
            raise ValueError("trajectory requires exact external raw-evidence custody")


class TrajectoryOwnerRecordReader(Protocol):
    def read_compiled(self, identity: ObjectIdentity) -> CompiledDeliveryControllerStudy: ...
    def read_tick(self, identity: ObjectIdentity) -> DeliveryControllerTickReceipt: ...
    def read_link(self, identity: ObjectIdentity) -> TrajectoryCallbackLink: ...


@dataclass(frozen=True, slots=True)
class TrajectoryUnitEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/trajectory-unit-evaluation'
    evaluation_id: str
    root: str
    plan: ObjectIdentity
    bundle: ObjectIdentity
    independent_unit_count: int
    committed_callbacks: int
    A: bool
    J: bool
    C: bool
    completion_ratio: Decimal | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        validate_stable_id(self.root, field_name="root")
        if (
            type(self.independent_unit_count) is not int
            or self.independent_unit_count != 1
            or any(type(v) is not bool for v in (self.A, self.J, self.C))
            or self.C != (self.A and self.J)
            or type(self.committed_callbacks) is not int
            or not 0 <= self.committed_callbacks <= 10000
            or self.plan.object_schema != TrajectoryControllerEvaluationPlan.SCHEMA
            or self.bundle.object_schema != RevealedTrajectoryBundle.SCHEMA
        ):
            raise ValueError("trajectory conjunction or independent n is invalid")
        if self.completion_ratio is not None and (
            not self.completion_ratio.is_finite() or self.completion_ratio < 0
        ):
            raise ValueError("trajectory completion ratio is not finite/nonnegative")
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("trajectory reasons must be canonical")


def evaluate_trajectory(
    plan: TrajectoryControllerEvaluationPlan,
    bundle: RevealedTrajectoryBundle,
    reader: TrajectoryOwnerRecordReader,
) -> TrajectoryUnitEvaluation:
    """Noncompensating existing-owner reduction of immutable callback evidence."""
    identity = ObjectIdentity.from_record(plan.evaluation_plan_id, plan)
    if (
        bundle.plan != identity
        or bundle.root not in plan.roots
        or len(bundle.callbacks) > plan.callback_count
    ):
        raise ValueError("trajectory changed its frozen plan/root/callback census")
    complete = bundle.native_complete and len(bundle.callbacks) == plan.callback_count
    reasons = set() if complete else {"INCOMPLETE_NATIVE_TRAJECTORY"}
    adequate, task = complete and bundle.raw_path_valid, complete
    if not bundle.raw_path_valid:
        reasons.add("RAW_CAUSAL_OR_NUMERICAL_DELIVERY_INVALID")
    tolerances = {v.value_id: v for v in plan.numerical_tolerances}
    widths = {v.value_id: v for v in plan.maximum_halfwidths}
    final = {}
    committed = 0
    previous_tick = None
    for child in bundle.callbacks:
        compiled, tick = reader.read_compiled(child.compiled), reader.read_tick(child.tick)
        link = reader.read_link(child.link)
        if (
            ObjectIdentity.from_record(compiled.compiled_study_id, compiled) != child.compiled
            or ObjectIdentity.from_record(tick.tick_id, tick) != child.tick
            or tick.compiled_study != child.compiled
            or compiled.study.instance_binding != tick.commitment.instance_binding
            or tick.commitment.instance_binding.frozen_recipe != plan.frozen_consumer
            or compiled.study.prospective_evaluation != identity
            or tick.commitment.commitment_coordinate.coordinate
            != child.index * plan.callback_seconds
            or tick.commitment.observation.coordinate.coordinate
            != child.index * plan.callback_seconds
        ):
            raise ValueError("trajectory substitutes a callback, consumer, clock or owner record")
        inputs = tick.commitment.observation.input_artifacts
        if (
            ObjectIdentity.from_record(link.link_id, link) != child.link
            or (link.root, link.index, link.previous_tick)
            != (bundle.root, child.index, previous_tick)
            or tick.commitment.observation.independent_unit_id != bundle.root
            or link.causal_input not in inputs
            or not any(
                a.artifact_id == link.link_id
                and a.sha256 == link.fingerprint()
                and a.payload_schema == link.SCHEMA
                for a in inputs
            )
        ):
            raise ValueError(
                "trajectory substitutes its independent root or prospective predecessor link"
            )
        previous_tick = child.tick
        delivered = (
            tick.disposition is TickDisposition.ACTION_DELIVERED and tick.delivery_trace.exact
        )
        if not delivered:
            adequate = False
            reasons.add("NONATTEMPT_OR_INEXACT_DELIVERY")
            if child.index != len(bundle.callbacks) - 1:
                raise ValueError("trajectory resumed after refusal or failed delivery")
        else:
            committed += 1
        nominal_present = bool(child.labels) and set(v.value_id for v in child.labels[0]) == set(
            tolerances
        )
        if nominal_present:
            nominal = {v.value_id: v for v in child.labels[0]}
            if any(nominal[key].unit != v.unit for key, v in tolerances.items()):
                raise ValueError("trajectory nominal measurement units differ")
            if float(nominal[plan.path_upper.value_id].value) > float(plan.path_upper.value):
                task = False
                reasons.add("NATIVE_PATH_LIMIT")
            final = nominal
        else:
            task = False
            reasons.add("MISSING_NOMINAL_MEASUREMENT")
        if len(child.labels) != 2 or any(
            set(v.value_id for v in view) != set(tolerances) for view in child.labels
        ):
            adequate = False
            reasons.add("MISSING_ASSIGNED_MEASUREMENT")
            continue
        labels = tuple({v.value_id: v for v in view} for view in child.labels)
        if any(
            labels[i][key].unit != tolerance.unit
            for key, tolerance in tolerances.items()
            for i in (0, 1)
        ):
            raise ValueError("trajectory raw measurement units differ")
        if any(
            abs(float(labels[0][key].value) - float(labels[1][key].value)) > float(tolerance.value)
            for key, tolerance in tolerances.items()
        ):
            adequate = False
            reasons.add("NUMERICAL_INVALID")
        selection = tick.commitment.selected_cell_action
        if selection is not None:
            matches = [
                r.request.response_set
                for r in compiled.study.admission.corpus.reachability_receipts
                if r.planned_coordinate.action_fibre.object_id
                == selection.action_binding.action_binding_id
            ]
            if len(matches) != 1:
                raise ValueError("selected callback lacks its unique owner prediction set")
            bounds = matches[0].bounds
            for i, view in enumerate(plan.numerical_views):
                by_quantity = {
                    b.coordinate.quantity_id: b for b in bounds if b.qualification_view_id == view
                }
                if set(by_quantity) != set(widths):
                    adequate = False
                    reasons.add("MISSING_CHOSEN_PREDICTION")
                    continue
                for key, maximum in widths.items():
                    bound = by_quantity[key]
                    low, high = bound.expanded_interval
                    covered = (
                        abs(float(labels[i][key].value) - float(bound.absolute_mean))
                        <= float(bound.calibrated_halfwidth)
                        if isinstance(bound, FiniteBinary64ResponseBound)
                        else low <= labels[i][key].value <= high
                    )
                    if not covered:
                        adequate = False
                        reasons.add("COVERAGE")
                    halfwidth = (
                        bound.calibrated_halfwidth
                        if isinstance(bound, FiniteBinary64ResponseBound)
                        else (high - low) / 2
                    )
                    if halfwidth > maximum.value or bound.coordinate.unit != maximum.unit:
                        adequate = False
                        reasons.add("WIDTH_OR_UNIT")
        else:
            adequate = False
    if not final or any(
        key.value_id not in final
        or final[key.value_id].unit != key.unit
        or float(final[key.value_id].value) < float(key.value)
        for key in plan.final_lower
    ):
        task = False
        reasons.add("NATIVE_FINAL_TARGET")
    ratio = (
        Decimal(repr(float(bundle.completion_seconds) / float(bundle.reference_completion_seconds)))
        if bundle.completion_seconds is not None
        and bundle.reference_completion_seconds is not None
        and bundle.reference_completion_seconds > 0
        else None
    )
    if ratio is None or ratio > plan.maximum_completion_ratio:
        task = False
        reasons.add("NATIVE_COMPLETION_TIME")
    return TrajectoryUnitEvaluation(
        f"trajectory.{bundle.root}",
        bundle.root,
        identity,
        ObjectIdentity.from_record(bundle.bundle_id, bundle),
        1,
        committed,
        adequate,
        task,
        adequate and task,
        ratio,
        tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class TrajectoryCohortEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/trajectory-cohort-evaluation'
    plan: TrajectoryControllerEvaluationPlan
    units: tuple[TrajectoryUnitEvaluation, ...]
    independent_unit_count: int
    A_count: int
    J_count: int
    C_count: int
    panel_mean_ratio: Decimal | None
    panel_mean_passed: bool

    def __post_init__(self) -> None:
        identity = ObjectIdentity.from_record(self.plan.evaluation_plan_id, self.plan)
        if tuple(u.root for u in self.units) != self.plan.roots or any(
            u.plan != identity for u in self.units
        ):
            raise ValueError("trajectory cohort changed its complete assigned-root census")
        if (self.independent_unit_count, self.A_count, self.J_count, self.C_count) != (
            len(self.units),
            sum(u.A for u in self.units),
            sum(u.J for u in self.units),
            sum(u.C for u in self.units),
        ):
            raise ValueError("trajectory cohort counts are not derived")
        ratios = tuple(u.completion_ratio for u in self.units)
        expected = (
            sum((r for r in ratios if r is not None), Decimal(0)) / len(ratios)
            if all(r is not None for r in ratios)
            else None
        )
        if self.panel_mean_ratio != expected or self.panel_mean_passed != (
            expected is not None and expected <= self.plan.panel_mean_completion_ratio
        ):
            raise ValueError("trajectory panel mean must remain separate and fully observed")
