"Native source qualification carrier and substrate declarations, without source contact."

from dataclasses import dataclass
from decimal import Decimal

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.observation_order import ObservationPhysicalUnit
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment, SourceQualificationSegment, SourceQualificationView
from empirical_lawhood.runtime.response_experiment_ports import NativeActionContract, NativeClockContract, NativeInteractionKind, NativeReceiverContract, ResponseSubstrateBinding
from empirical_lawhood.runtime.source_qualification import FreshSourceQualificationSubstrateBinding
from empirical_lawhood.adapters.methods.prepared_response.qualification_records import PreparedResponseSourceQualificationViewObservation
from .contracts import CAMPAIGN, PreparedNativeSpec
from .native_tasks import PreparedNativeInvocation, prepared_static_native_invocations
from .source_outputs import PreparedNativeTaskResult
from .extension_bundle import SOURCE_QUALIFICATION_NAMESPACE, SOURCE_QUALIFICATION_SOURCE_CAPABILITY


SOURCE_QUALIFICATION_CLOCK = f"{SOURCE_QUALIFICATION_NAMESPACE}.reference-clock"
SOURCE_QUALIFICATION_FRAME = f"{SOURCE_QUALIFICATION_NAMESPACE}.frozen-preparent-two-port-frame"
SOURCE_QUALIFICATION_RECEIVERS = (f"{SOURCE_QUALIFICATION_NAMESPACE}.receiver.m1", f"{SOURCE_QUALIFICATION_NAMESPACE}.receiver.m2")


@dataclass(frozen=True, slots=True)
class PreparedResponseSourceQualificationNativeDeclarations:
    units: tuple[ObservationPhysicalUnit, ...]
    segments: tuple[SourceQualificationSegment, ...]
    views: tuple[SourceQualificationView, ...]


def prepared_response_source_qualification_native_declarations(spec: PreparedNativeSpec) -> PreparedResponseSourceQualificationNativeDeclarations:
    if spec.stage != 'qualification':
        raise ValueError("prepared source qualification carrier cannot reinterpret another source stage")
    units = tuple(
        ObservationPhysicalUnit(
            root.physical_unit_id,
            f"{CAMPAIGN}.coordinate.{root.context}.t{root.landmark}",
            f"{root.physical_unit_id}.instance",
            f"{CAMPAIGN}.family.{root.context}",
            root.fingerprint(),
            None,
        )
        for root in spec.roots
    )
    segments = []
    tasks = prepared_static_native_invocations(spec)
    for task in tasks:
        start, end = (
            (0, task.root.landmark)
            if task.phase == "prefix"
            else (task.root.landmark, task.root.handoff)
            if task.phase == "parent"
            else (task.root.handoff, task.root.handoff + 320)
        )
        actions = (
            ()
            if task.phase == "prefix"
            else tuple(
                f"{task.task_id}.{kind}.{axis}" for axis in ("m1", "m2") for kind in ("force",)
            )
            if task.phase == "future"
            else (f"{task.task_id}.coupling.x", f"{task.task_id}.coupling.y")
        )
        segments.append(
            SourceQualificationSegment(
                task.task_id,
                f"group.{task.task_id}",
                task.root.physical_unit_id,
                None if not task.dependency_task_ids else task.dependency_task_ids[0],
                SOURCE_QUALIFICATION_CLOCK,
                Decimal(start),
                Decimal(end),
                tuple(f"{task.root.root_id}.project.r{r}" for r in (1, 2)),
                actions,
            )
        )
    views = tuple(
        SourceQualificationView(
            f"{root.root_id}.project.r{r}",
            root.physical_unit_id,
            spec.numerical_views[r - 1].view_id,
            tuple(task.task_id for task in tasks if task.root == root),
        )
        for root in spec.roots
        for r in (1, 2)
    )
    return PreparedResponseSourceQualificationNativeDeclarations(units, tuple(segments), views)


def prepared_response_source_qualification_substrate(
    spec: PreparedNativeSpec,
    carrier: FreshSourceQualificationExperiment,
    installed_binding: ObjectIdentity,
) -> FreshSourceQualificationSubstrateBinding:
    source = ObjectIdentity.from_record(spec.spec_id, spec)
    substrate = ResponseSubstrateBinding(
        f"{SOURCE_QUALIFICATION_NAMESPACE}.native-binding",
        f"{CAMPAIGN}.six-matrix-q2-medium",
        NativeInteractionKind.INTERACTIVE_EXECUTION,
        SOURCE_QUALIFICATION_SOURCE_CAPABILITY.capability_key,
        SOURCE_QUALIFICATION_SOURCE_CAPABILITY.capability_version,
        SOURCE_QUALIFICATION_SOURCE_CAPABILITY.implementation_sha256,
        installed_binding,
        PreparedNativeSpec.SCHEMA,
        PreparedNativeInvocation.SCHEMA,
        None,
        PreparedNativeTaskResult.SCHEMA,
        PreparedResponseSourceQualificationViewObservation.SCHEMA,
        source,
        tuple(
            NativeReceiverContract(
                receiver,
                f"{SOURCE_QUALIFICATION_NAMESPACE}.quantity.receiver.m{index}",
                "hilbert-schmidt-native",
                SOURCE_QUALIFICATION_FRAME,
                f"{CAMPAIGN}.frozen-primary-port.m{index}",
                (SOURCE_QUALIFICATION_CLOCK,),
            )
            for index, receiver in enumerate(SOURCE_QUALIFICATION_RECEIVERS, 1)
        ),
        (NativeClockContract(SOURCE_QUALIFICATION_CLOCK, "reference-tick", SOURCE_QUALIFICATION_FRAME),),
        NativeActionContract(
            f"{SOURCE_QUALIFICATION_NAMESPACE}.quantity.two-port-action", True, True, True, True, "HOLD"
        ),
        False,
        False,
        False,
        OutcomeAccess.OUTCOME_BLIND,
    )
    return FreshSourceQualificationSubstrateBinding(
        f"{SOURCE_QUALIFICATION_NAMESPACE}.substrate",
        ObjectIdentity.from_record(carrier.extension_set_id, carrier),
        substrate,
        source,
    )
