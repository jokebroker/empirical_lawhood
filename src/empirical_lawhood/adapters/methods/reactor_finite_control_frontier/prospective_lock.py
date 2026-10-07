"""All context/request commitments persist before any local source branch."""

from dataclasses import dataclass
from collections.abc import Iterator
from decimal import Decimal as D
from typing import ClassVar, Any

from empirical_lawhood.adapters.control.prepared_forecast import PreparedForecastLock, freeze_forecast_request
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.receiver_conditioned_io.finite_action_mpc import FiniteMPCProposal
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.words import PulseProjection
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.finite_response_geometry import FiniteResponseSet, FiniteTaskFunctionalSpec
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.canonical_record_archive import CanonicalRecordArchive
from empirical_lawhood.runtime.controller_compiler import CompiledDeliveryControllerStudy
from empirical_lawhood.runtime.controller_runtime import DeliveryControllerDecisionCommitment
from empirical_lawhood.adapters.control.publication import ControlPublisher, ControlPublicationSession
from .control_owner import assess_candidate, rank_and_commit
from .causal import causal_operands
from .law_terminal import FrontierLaws
from .prospective_plan import ReactorFiniteControlFrontierProspectivePlan
from .records import FrontierPreparation
from .selection import FrontierPredictionSeal, FrontierUseRequest, eligible_el_pulses


class UndeliverableSession:
    def advance(self, command: Any) -> Any:
        raise AssertionError("causal controller preparation attempted native delivery")


@dataclass(frozen=True, slots=True)
class FrontierFrozenUse(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-frozen-use'
    request: FrontierUseRequest
    projection: PulseProjection | None
    compiled: CompiledDeliveryControllerStudy | None
    commitment: DeliveryControllerDecisionCommitment | None
    task: FiniteTaskFunctionalSpec | None
    forecasts: tuple[FiniteResponseSet, ...]
    proposal: FiniteMPCProposal | None
    lock: PreparedForecastLock | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.projection is None:
            if (
                any(v is not None for v in (self.compiled, self.commitment, self.task, self.lock))
                or self.forecasts
                or not self.reasons
            ):
                raise ValueError("nonattempt invented an owner commitment")
        elif (
            any(
                v is None
                for v in (self.compiled, self.commitment, self.task, self.proposal, self.lock)
            )
            or len(self.forecasts) != 1
            or self.reasons
            or self.commitment is None
            or self.commitment.action_binding is None
            or self.commitment.action_binding.action_word != self.projection.word
        ):
            raise ValueError("frozen use lost its exact chosen word or prepared-owner lock")


@dataclass(frozen=True, slots=True)
class FrontierFrozenRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-frozen-root'
    root: str
    preparation: ObjectIdentity
    prediction_seal: ObjectIdentity
    plan: ObjectIdentity
    use_archives: tuple[CanonicalRecordArchive, ...]

    def __post_init__(self) -> None:
        if len(self.use_archives) > 24 or any(
            a.subject.object_schema != FrontierFrozenUse.SCHEMA
            or not a.subject.object_id.startswith(f"{self.root}.")
            for a in self.use_archives
        ):
            raise ValueError("frozen request archive changed its root or owner schema")

    @property
    def uses(self) -> Iterator[FrontierFrozenUse]:
        """Authenticate and decode one unchanged owner record at a time."""
        for archive in self.use_archives:
            use = decode_canonical_bytes(
                archive.unpack(), FrontierFrozenUse, maximum_bytes=archive.decoded_bytes
            )
            if archive.subject.object_id != f"{self.root}.{use.request.request_id}.frozen-use":
                raise ValueError("frozen request archive substituted its exact request")
            yield use

    @property
    def record_id(self) -> str:
        return f"{self.root}.frontier-frozen-root"


def freeze_root(
    *,
    laws: FrontierLaws,
    preparation: FrontierPreparation,
    seal: FrontierPredictionSeal,
    plan: ReactorFiniteControlFrontierProspectivePlan,
    publisher: ControlPublisher,
    reader: CandidatePayloadReader,
    control: Any,
    causal_receipt: CanonicalTaskReceipt,
    causal_artifact: ArtifactIdentity,
) -> FrontierFrozenRoot:
    parent = ObjectIdentity.from_record(preparation.record_id, preparation)
    if (
        seal.preparation != parent
        or seal.parent_laws != ObjectIdentity.from_record(laws.record_id, laws)
        or any(seal.development != r.payload.development for r in laws.rows)
        or plan.laws != seal.parent_laws
        or causal_artifact.sha256 != preparation.fingerprint()
    ):
        raise ValueError("Controller use lock changed its sealed parent or qualification source")
    publisher = ControlPublicationSession(publisher, maximum_records=4096)
    uses = []
    store = control.open_prepared_store()
    instant = control.freeze_clock(preparation.root)
    for request, expected in seal.primary:
        if expected is None:
            uses.append(
                FrontierFrozenUse(
                    request, None, None, None, None, (), None, None, ("CAUSAL_NONATTEMPT",)
                )
            )
            continue
        ca = next(c for c in preparation.contexts if c.context == request.context)
        bundle = next(p for p in plan.requests if p.request == request).unpack()
        evaluation = bundle.plan
        if evaluation is None or not next(
            yes for r, yes, _ in bundle.assigned if r == preparation.root
        ):
            raise ValueError("admitted causal choice is missing from its prepared census")
        operands = causal_operands(ca, laws.rows[0].payload.prepared_domain)
        eligible = eligible_el_pulses(
            bounds=tuple(r.payload.bound for r in laws.rows),
            context=request.context,
            horizon=request.horizon_s,
            request=request.request_K,
            budget=request.budget_kg,
            observed_temperature=None
            if operands is None
            else D(repr(operands.observation.temperature)),
            masses={p: mass for c, p, mass, _ in seal.masses if c == request.context},
            qualified=laws.qualified,
        )
        if not eligible or eligible[0] != expected:
            raise ValueError("Admission necessary causal filters disagree with the sealed selector")
        candidates = tuple(
            assess_candidate(
                laws=laws,
                report=report,
                causal=ca,
                request=request,
                publisher=publisher,
                reader=reader,
                session=UndeliverableSession(),
                authority=control.authority,
                resources=control.resources,
                evaluation=evaluation,
            )
            for report in laws.rows
            if report.payload.bound.coordinate.context == request.context
            and report.payload.bound.coordinate.horizon_s == request.horizon_s
            and report.payload.bound.coordinate.coordinate_id in laws.qualified
            and report.payload.bound.coordinate.pulse in eligible
        )
        proposal, prepared = rank_and_commit(candidates, publisher)
        if prepared is None or prepared.candidate.context.word.pulse != expected:
            raise ValueError("independent finite admission owners disagree with the sealed causal choice")
        candidate = prepared.candidate
        lock = freeze_forecast_request(
            prefix_id=f"{preparation.root}.{request.request_id}.prepared",
            policy_id=f"el.{request.request_id}",
            root=preparation.root,
            compiled=prepared.compiled,
            commitment=prepared.commitment,
            task=candidate.task,
            audit_forecasts=candidate.forecasts,
            checkpoint=parent,
            causal_receipt=causal_receipt,
            causal_artifact=causal_artifact,
            issued_study=control.issued_study,
            evaluation=evaluation,
            store=store,
            occurred_at_utc=instant,
        )
        uses.append(
            FrontierFrozenUse(
                request,
                candidate.context.word,
                prepared.compiled,
                prepared.commitment,
                candidate.task,
                candidate.forecasts,
                proposal,
                lock,
                (),
            )
        )
    return FrontierFrozenRoot(
        preparation.root,
        parent,
        ObjectIdentity.from_record(seal.record_id, seal),
        ObjectIdentity.from_record(plan.record_id, plan),
        tuple(
            CanonicalRecordArchive.pack(
                f"{preparation.root}.{u.request.request_id}.frozen-use", u
            )
            for u in uses
        ),
    )
