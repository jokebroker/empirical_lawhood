"Exact finite response-law native clocks, physical units and views for the shared carrier."

from dataclasses import dataclass, replace
from decimal import Decimal
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.source_qualification import PredecessorBoundSourceQualificationSubstrateBinding, QualifiedSourceUseSubstrateBinding, ProspectiveRetainedSourceUseBinding
from empirical_lawhood.planning.source_qualification import QualifiedSourceUseExperiment, ProspectiveRetainedSourceUse

from empirical_lawhood.planning.observation_order import ObservationPhysicalUnit
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment, SourceQualificationSegment, SourceQualificationView
from empirical_lawhood.adapters.methods.finite_response_law.science import PROGRAMME
from empirical_lawhood.adapters.simulators.prepared_response.contracts import prepared_numerical_view
from .contracts import FiniteResponseLawNativeConfig, native_invocations

Q_CLOCK = f"{PROGRAMME}.reference-clock"
Q_FRAME = f"{PROGRAMME}.frozen-preparent-two-port-frame"
Q_RECEIVERS = (f"{PROGRAMME}.receiver.m1", f"{PROGRAMME}.receiver.m2")


@dataclass(frozen=True, slots=True)
class FiniteResponseLawNativeDeclarations:
    units: tuple[ObservationPhysicalUnit, ...]
    segments: tuple[SourceQualificationSegment, ...]
    views: tuple[SourceQualificationView, ...]


def native_declarations(config: FiniteResponseLawNativeConfig) -> FiniteResponseLawNativeDeclarations:
    tasks = native_invocations(config)
    units = tuple(
        ObservationPhysicalUnit(
            root.physical_unit_id,
            f"{PROGRAMME}.coordinate.prepared.t4096",
            f"{root.physical_unit_id}.instance",
            f"{PROGRAMME}.family.prepared",
            root.fingerprint() if root.retained_root is None else root.retained_root.fingerprint(),
            None,
        )
        for root in config.roots
    )
    segments = []
    for task in tasks:
        actions = (
            ()
            if task.phase == "prefix"
            else tuple(
                f"{task.task_id}.{kind}.{axis}"
                for kind, axis in (("force", "m1"), ("force", "m2"))
                if task.phase == "future"
            )
        )
        if task.phase == "parent":
            actions = tuple(f"{task.task_id}.coupling.{axis}" for axis in ("x", "y"))
        segments.append(
            SourceQualificationSegment(
                task.task_id,
                f"group.{task.task_id}",
                task.root.physical_unit_id,
                task.predecessor_segment_id,
                Q_CLOCK,
                Decimal(task.clocks[0]),
                Decimal(task.clocks[1]),
                tuple(f"{task.root.root_id}.flh-project.r{r}" for r in (1, 2)),
                actions,
            )
        )
    views = tuple(
        SourceQualificationView(
            f"{root.root_id}.flh-project.r{r}",
            root.physical_unit_id,
            prepared_numerical_view(r).view_id,
            tuple(t.task_id for t in tasks if t.root == root),
        )
        for root in config.roots
        for r in (1, 2)
    )
    return FiniteResponseLawNativeDeclarations(units, tuple(segments), views)


def substrate_binding(
    config: FiniteResponseLawNativeConfig,
    carrier: 'PredecessorBoundSourceQualificationExperiment',
    installed_binding: "ObjectIdentity",
) -> 'PredecessorBoundSourceQualificationSubstrateBinding':
    from empirical_lawhood.kernel.evidence import OutcomeAccess
    from empirical_lawhood.runtime.response_experiment_ports import NativeActionContract, NativeClockContract, NativeInteractionKind, NativeReceiverContract, ResponseSubstrateBinding
    from empirical_lawhood.adapters.methods.finite_response_law.native_records import native_method_types
    from .protocol import native_capabilities
    from .provider import native_result_type

    source = ObjectIdentity.from_record(config.spec_id, config)
    source_capability, _, _ = native_capabilities(config)
    _, _, _, view_type, _ = native_method_types(config)
    invocation_type = type(native_invocations(config)[0])
    substrate = ResponseSubstrateBinding(
        f"{PROGRAMME}.native-binding",
        f"{PROGRAMME}.six-matrix-q2-medium",
        NativeInteractionKind.INTERACTIVE_EXECUTION,
        source_capability.capability_key,
        source_capability.capability_version,
        source_capability.implementation_sha256,
        installed_binding,
        config.SCHEMA,
        invocation_type.SCHEMA,
        None,
        native_result_type(config).SCHEMA,
        view_type.SCHEMA,
        source,
        tuple(
            NativeReceiverContract(
                receiver,
                f"{PROGRAMME}.quantity.receiver.m{i}",
                "hilbert-schmidt-native",
                Q_FRAME,
                f"{PROGRAMME}.frozen-primary-port.m{i}",
                (Q_CLOCK,),
            )
            for i, receiver in enumerate(Q_RECEIVERS, 1)
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
    binding_type = (
        ProspectiveRetainedSourceUseBinding
        if isinstance(carrier, ProspectiveRetainedSourceUse)
        else QualifiedSourceUseSubstrateBinding
        if isinstance(carrier, QualifiedSourceUseExperiment)
        else PredecessorBoundSourceQualificationSubstrateBinding
    )
    if isinstance(carrier, ProspectiveRetainedSourceUse):
        from .evaluation_continuation.discovery import SOURCE_CAPABILITY as continuation_source

        substrate = replace(
            substrate,
            provider_key=continuation_source.capability_key,
            provider_version=continuation_source.capability_version,
            provider_implementation_sha256=continuation_source.implementation_sha256,
        )
    return binding_type(
        f"{PROGRAMME}.substrate",
        ObjectIdentity.from_record(carrier.extension_set_id, carrier),
        substrate,
        source,
    )
