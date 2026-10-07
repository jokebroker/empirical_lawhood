"causal response prediction native carrier declarations through the shared qualification topology."

from dataclasses import dataclass
from decimal import Decimal

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.observation_order import ObservationPhysicalUnit
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment, SourceQualificationSegment, SourceQualificationView
from empirical_lawhood.runtime.response_experiment_ports import NativeActionContract, NativeClockContract, NativeInteractionKind, NativeReceiverContract, ResponseSubstrateBinding
from empirical_lawhood.runtime.source_qualification import FreshSourceQualificationSubstrateBinding
from empirical_lawhood.adapters.methods.causal_response.records import CausalResponseViewObservation
from empirical_lawhood.adapters.simulators.prepared_response.native_tasks import PreparedNativeInvocation
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from .contracts import CausalResponseNativeConfig, native_invocations
from .extension_bundle import NAMESPACE, SOURCE_CAPABILITY

CAMPAIGN = NAMESPACE


REFERENCE_CLOCK = f"{NAMESPACE}.reference-clock"
PREPARENT_FRAME = f"{NAMESPACE}.frozen-preparent-two-port-frame"
NATIVE_RECEIVERS = (f"{NAMESPACE}.receiver.m1", f"{NAMESPACE}.receiver.m2")


@dataclass(frozen=True, slots=True)
class CausalResponseNativeDeclarations:
    units: tuple[ObservationPhysicalUnit, ...]
    segments: tuple[SourceQualificationSegment, ...]
    views: tuple[SourceQualificationView, ...]


def native_declarations(spec: CausalResponseNativeConfig) -> CausalResponseNativeDeclarations:
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
    tasks = native_invocations(spec)
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
                REFERENCE_CLOCK,
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
            spec.recipe.numerical_views[r - 1].view_id,
            tuple(task.task_id for task in tasks if task.root == root),
        )
        for root in spec.roots
        for r in (1, 2)
    )
    return CausalResponseNativeDeclarations(units, tuple(segments), views)


def substrate_binding(
    spec: CausalResponseNativeConfig,
    carrier: FreshSourceQualificationExperiment,
    installed_binding: ObjectIdentity,
) -> FreshSourceQualificationSubstrateBinding:
    source = ObjectIdentity.from_record(spec.spec_id, spec)
    substrate = ResponseSubstrateBinding(
        f"{NAMESPACE}.native-binding",
        f"{CAMPAIGN}.six-matrix-q2-medium",
        NativeInteractionKind.INTERACTIVE_EXECUTION,
        SOURCE_CAPABILITY.capability_key,
        SOURCE_CAPABILITY.capability_version,
        SOURCE_CAPABILITY.implementation_sha256,
        installed_binding,
        CausalResponseNativeConfig.SCHEMA,
        PreparedNativeInvocation.SCHEMA,
        None,
        PreparedNativeTaskResult.SCHEMA,
        CausalResponseViewObservation.SCHEMA,
        source,
        tuple(
            NativeReceiverContract(
                receiver,
                f"{NAMESPACE}.quantity.receiver.m{index}",
                "hilbert-schmidt-native",
                PREPARENT_FRAME,
                f"{CAMPAIGN}.frozen-primary-port.m{index}",
                (REFERENCE_CLOCK,),
            )
            for index, receiver in enumerate(NATIVE_RECEIVERS, 1)
        ),
        (NativeClockContract(REFERENCE_CLOCK, "reference-tick", PREPARENT_FRAME),),
        NativeActionContract(
            f"{NAMESPACE}.quantity.two-port-action", True, True, True, True, "HOLD"
        ),
        False,
        False,
        False,
        OutcomeAccess.OUTCOME_BLIND,
    )
    return FreshSourceQualificationSubstrateBinding(
        f"{NAMESPACE}.substrate",
        ObjectIdentity.from_record(carrier.extension_set_id, carrier),
        substrate,
        source,
    )
