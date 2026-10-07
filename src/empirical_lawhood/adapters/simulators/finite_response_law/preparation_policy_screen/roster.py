"""Exact preparation-policy development units, +400/+192 segments, and paired-view projections."""

from dataclasses import dataclass
from decimal import Decimal

from empirical_lawhood.adapters.methods.finite_response_law.science import PROGRAMME
from empirical_lawhood.adapters.methods.finite_response_law.preparation_policy_projection import FiniteResponseLawPreparationPolicyRootPanel
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import FiniteResponseLawPreparationPolicyNativeConfig, preparation_policy_native_invocations
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_source_outputs import FiniteResponseLawPreparationPolicyNativeTaskResult
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.observation_order import ObservationPhysicalUnit
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment, SourceQualificationSegment, SourceQualificationView
from empirical_lawhood.runtime.response_experiment_ports import NativeActionContract, NativeClockContract, NativeInteractionKind, NativeReceiverContract, ResponseSubstrateBinding
from empirical_lawhood.runtime.source_qualification import PredecessorBoundSourceQualificationSubstrateBinding

from .discovery import SOURCE_CAPABILITY

Q_CLOCK = f"{PROGRAMME}.reference-clock"
Q_FRAME = f"{PROGRAMME}.frozen-preparent-two-port-frame"
Q_RECEIVERS = (f"{PROGRAMME}.receiver.m1", f"{PROGRAMME}.receiver.m2")


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyDeclarations:
    units: tuple[ObservationPhysicalUnit, ...]
    segments: tuple[SourceQualificationSegment, ...]
    views: tuple[SourceQualificationView, ...]


def preparation_policy_declarations(config: FiniteResponseLawPreparationPolicyNativeConfig) -> FiniteResponseLawPreparationPolicyDeclarations:
    tasks = preparation_policy_native_invocations(config)
    units = tuple(sorted((
        ObservationPhysicalUnit(
            root.physical_unit_id,
            f"{PROGRAMME}.coordinate.prepared.t4096",
            f"{root.physical_unit_id}.instance",
            f"{PROGRAMME}.family.prepared",
            root.retained_prefix.fingerprint() if root.retained_prefix is not None else "",
            None,
        )
        for root in config.roots
    ), key=lambda value: value.physical_independent_unit_id))
    segments = tuple(
        SourceQualificationSegment(
            task.task_id,
            f"group.{task.task_id}",
            task.root.physical_unit_id,
            task.predecessor_segment_id,
            Q_CLOCK,
            Decimal(task.clocks[0]),
            Decimal(task.clocks[1]),
            (f"{task.root.stage_unit}.project",),
            tuple(
                f"{task.task_id}.{kind}.{axis}"
                for kind, axis in (
                    (("force", "m1"), ("force", "m2"))
                    if task.phase == "future"
                    else (("coupling", "x"), ("coupling", "y"))
                )
            ),
        )
        for task in tasks
    )
    views = tuple(
        SourceQualificationView(
            f"{root.stage_unit}.project",
            root.physical_unit_id,
            f"{PROGRAMME}.paired-primary-r1-and-half-r2",
            tuple(task.task_id for task in tasks if task.root == root),
        )
        for root in config.roots
    )
    return FiniteResponseLawPreparationPolicyDeclarations(units, segments, views)


def preparation_policy_substrate_binding(
    config: FiniteResponseLawPreparationPolicyNativeConfig,
    carrier: PredecessorBoundSourceQualificationExperiment,
    installed_binding: ObjectIdentity,
) -> PredecessorBoundSourceQualificationSubstrateBinding:
    source = ObjectIdentity.from_record(config.spec_id, config)
    substrate = ResponseSubstrateBinding(
        f"{PROGRAMME}.preparation-policy.native-binding",
        f"{PROGRAMME}.six-matrix-q2-medium",
        NativeInteractionKind.INTERACTIVE_EXECUTION,
        SOURCE_CAPABILITY.capability_key,
        SOURCE_CAPABILITY.capability_version,
        SOURCE_CAPABILITY.implementation_sha256,
        installed_binding,
        config.SCHEMA,
        preparation_policy_native_invocations(config)[0].SCHEMA,
        None,
        FiniteResponseLawPreparationPolicyNativeTaskResult.SCHEMA,
        FiniteResponseLawPreparationPolicyRootPanel.SCHEMA,
        source,
        tuple(
            NativeReceiverContract(
                receiver,
                f"{PROGRAMME}.quantity.receiver.m{index}",
                "hilbert-schmidt-native",
                Q_FRAME,
                f"{PROGRAMME}.frozen-primary-port.m{index}",
                (Q_CLOCK,),
            )
            for index, receiver in enumerate(Q_RECEIVERS, 1)
        ),
        (NativeClockContract(Q_CLOCK, "reference-tick", Q_FRAME),),
        NativeActionContract(
            f"{PROGRAMME}.quantity.two-port-action", True, True, True, True, "HOLD"
        ),
        False,
        False,
        False,
        OutcomeAccess.OUTCOME_BLIND,
    )
    return PredecessorBoundSourceQualificationSubstrateBinding(
        f"{PROGRAMME}.preparation-policy.substrate",
        ObjectIdentity.from_record(carrier.extension_set_id, carrier),
        substrate,
        source,
    )
