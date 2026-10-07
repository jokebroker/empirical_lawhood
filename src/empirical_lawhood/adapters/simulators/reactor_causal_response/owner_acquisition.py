"""Effect-bound callback sequence; records are durable before native advance."""

from dataclasses import dataclass, field
from time import process_time
from typing import Any
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.runtime.controller_runtime import TickDisposition
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.reactor_causal_response.control_owner import CallbackRecordPublisher, prepare_callback, trajectory_plan
from empirical_lawhood.adapters.methods.reactor_causal_response.terminal import EmpiricalQualificationResult
from empirical_lawhood.adapters.methods.reactor_causal_response.controller import NumericalController
from empirical_lawhood.adapters.simulators.reactor_prefix_response.contracts import ReactorDelivery, ReactorFinalDelivery
from .acquisition import NativeController


@dataclass
class EmpiricalCallbackSequence:
    report: EmpiricalQualificationResult
    root: str
    publisher: CallbackRecordPublisher
    reader: CandidatePayloadReader
    calibration_artifact: ArtifactIdentity
    authority: ObjectIdentity
    resources: ObjectIdentity
    children: list[tuple[ObjectIdentity, ObjectIdentity, ObjectIdentity]] = field(
        default_factory=list
    )
    owner_cpu: list[float] = field(default_factory=list)
    failed: bool = False

    def advance(
        self, controller: NativeController, previous: tuple[float, float], session: Any
    ) -> ReactorDelivery | ReactorFinalDelivery | None:
        if (
            not isinstance(controller, NumericalController)
            or len(self.children) != len(controller.history) - 1
        ):
            raise ValueError("owner sequence changed controller or callback prefix")
        start = process_time()
        prepared = prepare_callback(
            report=self.report,
            root=self.root,
            observation=controller.history[-1],
            previous=previous,
            decision=controller.decisions[-1],
            previous_tick=self.children[-1][1] if self.children else None,
            publisher=self.publisher,
            reader=self.reader,
            session=session,
            calibration_artifact=self.calibration_artifact,
            authority=self.authority,
            resources=self.resources,
            evaluation=trajectory_plan(self.report),
        )
        tick = prepared.deliver(self.publisher)
        self.owner_cpu.append(process_time() - start)
        self.children.append(
            (
                ObjectIdentity.from_record(
                    prepared.compiled.compiled_study_id, prepared.compiled
                ),
                ObjectIdentity.from_record(tick.tick_id, tick),
                ObjectIdentity.from_record(prepared.link.link_id, prepared.link),
            )
        )
        self.failed = (
            tick.disposition is not TickDisposition.ACTION_DELIVERED
            or not tick.delivery_trace.exact
        )
        return prepared.delivery.last_delivery
