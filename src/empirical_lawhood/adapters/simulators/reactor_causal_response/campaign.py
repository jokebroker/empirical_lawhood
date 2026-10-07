"""The preassigned nine-arm campaign, with primary callbacks owned before effect."""

from decimal import Decimal as D
import json
import resource
from time import monotonic
from typing import Callable, Protocol
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.controller_evaluation_trajectory import TrajectoryOwnerRecordReader
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalComparatorSources, TimedReactorConfirmationEnvelope, EmpiricalEpisodeTiming
from empirical_lawhood.adapters.methods.reactor_causal_response.experiment_records import EmpiricalDiscoveryEnvelope, EmpiricalQualificationTerminal, EmpiricalAcquisitionEnvelope
from empirical_lawhood.adapters.methods.reactor_causal_response.control_owner import CallbackRecordPublisher
from empirical_lawhood.adapters.methods.reactor_causal_response.control_services import implementation_payloads, implementation_configurations
from empirical_lawhood.adapters.methods.reactor_causal_response.comparators import DiagnosticController, published_controller
from empirical_lawhood.adapters.methods.reactor_causal_response.controller import NumericalController
from empirical_lawhood.adapters.methods.reactor_causal_response.serialization import read_fit
from empirical_lawhood.adapters.methods.reactor_causal_response.config import ARMS
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from .acquisition import acquire_episode, TapeController
from .owner_acquisition import EmpiricalCallbackSequence
from .interface import Actuator
from .design import scenario


class CallbackArchive(CallbackRecordPublisher, TrajectoryOwnerRecordReader, Protocol):
    """Authenticated external publication and same-identity replay, no simulator access."""


class ControlCustodyPort(Protocol):
    authority: ObjectIdentity
    resources: ObjectIdentity

    def open_control_store(self, root: str) -> CallbackArchive: ...


def prerequisite(qualification: EmpiricalQualificationTerminal) -> str | None:
    if qualification.prerequisite is not None:
        return qualification.prerequisite
    result = qualification.result
    if result is None or result.qualification.response_law is None or result.payload.q is None:
        return "LOCAL_LAW_NOT_SUPPORTED"
    if qualification.usefulness is None or not qualification.usefulness.useful:
        return "USEFULNESS_NOT_SUPPORTED"
    return None


def acquire_confirmation(
    source: ReactorBatchSource,
    comparators: EmpiricalComparatorSources,
    discovery: EmpiricalDiscoveryEnvelope,
    qualification: EmpiricalQualificationTerminal,
    ordinal: int,
    *,
    custody: ControlCustodyPort,
    reader: CandidatePayloadReader,
    progress: Callable[[int], None] | None = None,
) -> TimedReactorConfirmationEnvelope:
    assigned = scenario("confirmation", ordinal)
    qid = ObjectIdentity.from_record("reactor-empirical-qualification", qualification)
    did = ObjectIdentity.from_record("reactor-empirical-discovery", discovery)
    stop = prerequisite(qualification)
    if stop is not None:
        return TimedReactorConfirmationEnvelope(assigned.unit_id, qid, did, None, (), (), stop)
    report = qualification.result
    assert report is not None and report.payload.q is not None
    saved = json.loads(discovery.result_json)
    if discovery.recipe != qualification.recipe or saved["selected"] is None:
        raise ValueError("confirmation substitutes its frozen discovery/qualification parent")
    model = read_fit(json.loads(report.payload.model_json))
    if read_fit(saved["fits"][saved["selected"]]) != model:
        raise ValueError("confirmation nominee differs from qualified payload")
    q = float(report.payload.q)
    archive = custody.open_control_store(assigned.unit_id)
    calibration = archive.publish_record("reactor-empirical-calibration-operands", report.operands)
    for config in implementation_configurations():
        # These small role payloads have direct byte identities in their bindings.
        artifact = archive.publish_record(config.config_id, config)
        expected = next(
            b.reference.payload for b, _ in implementation_payloads() if b.role is config.role
        )
        if artifact != expected:
            raise ValueError("controller implementation payload publication differs")
    owner = EmpiricalCallbackSequence(
        report, assigned.unit_id, archive, reader, calibration, custody.authority, custody.resources
    )
    rotation = ordinal % len(ARMS)
    order = ARMS[rotation:] + ARMS[:rotation]
    episodes = []
    telemetry = {}
    done = 0

    def emit(k: int) -> None:
        if progress is not None:
            progress(done + k)

    for arm in order:
        if arm == "EL":
            controller = NumericalController(model, q, Actuator())
        elif arm in ("F0", "F1", "MARGIN", "FIXED"):
            rival = (
                read_fit(saved["fits"][saved["rivals"][int(arm[-1])]])
                if arm in ("F0", "F1")
                else model
            )
            controller = DiagnosticController(
                rival,
                q,
                Actuator(),
                arm,
                published_controller("SCHEDULED_BACKOFF_HALF", comparators.source("SCHEDULED_BACKOFF_HALF")) if arm == "FIXED" else None,
            )
        else:
            controller = published_controller(arm, comparators.source(arm))
        callback_wall: list[float] = []
        started = monotonic()
        episode = acquire_episode(
            source,
            assigned,
            arm,
            controller,
            owner=owner if arm == "EL" else None,
            progress=None if progress is None else emit,
            callback_wall=callback_wall,
        )
        telemetry[arm] = EmpiricalEpisodeTiming(
            arm,
            tuple(D(repr(t)) for t in callback_wall),
            D(repr(monotonic() - started)),
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        episodes.append(episode)
        done += len(episode.requests)
    nominal = next(e for e in episodes if e.episode == "EL")
    refined = acquire_episode(
        source,
        assigned,
        "EL",
        TapeController(nominal.requests),
        dt=0.5,
        progress=None if progress is None else emit,
    )
    episodes.append(refined)
    return TimedReactorConfirmationEnvelope(
        assigned.unit_id,
        qid,
        did,
        EmpiricalAcquisitionEnvelope(
            assigned.unit_id, "confirmation", tuple(e.envelope() for e in episodes)
        ),
        tuple(owner.children),
        tuple(D(repr(x)) for x in owner.owner_cpu),
        None,
        telemetry=tuple(telemetry[arm] for arm in ARMS),
    )
