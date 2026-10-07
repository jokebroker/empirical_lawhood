"Nonexecuting retained-parent check, not an admission or controller use finalizer.\n\nThis diagnostic exercises the existing law evaluator without fitting, publishing\nor changing the parent. A known missing forecast is a prerequisite nonattempt,\nnot a failed scientific control trial. A future eligible parent still needs the\nordinary admission, reachability, programme and prospective owners.\n"

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import ClassVar, cast

from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.law_evaluation import (
    FiniteActionLawEvaluator,
    LawEvaluationDisposition,
    LawEvaluationKind,
    LawEvaluationMode,
    LawEvaluationRequest,
    LawEvaluationResult,
    LawEvaluationService,
    LawEvaluatorImplementation,
    LawEvaluatorRegistration,
    LawEvaluatorRegistry,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.time import HorizonSpec
from .result import ReactorScienceResult


@dataclass(frozen=True, slots=True)
class ReactorAdmissionEntryDiagnostic(CanonicalRecord):
    "Exposed-parent interface evidence; grants no admission/controller use scientific status."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prefix-response/reactor-admission-entry-diagnostic'

    parent: ObjectIdentity
    qualified_horizon: HorizonSpec
    requested_horizon: HorizonSpec
    nontransported_properties: tuple[str, ...]
    local_requests: tuple[LawEvaluationRequest, ...]
    local_results: tuple[LawEvaluationResult, ...]
    sink_requests: tuple[LawEvaluationRequest, ...]
    sink_results: tuple[LawEvaluationResult, ...]
    horizon_refusal: str
    receiver_refusal: str

    def __post_init__(self) -> None:
        if not self.local_results or len(self.local_requests) != len(self.local_results):
            raise ValueError("entry diagnostic requires its complete local evaluator probes")
        if len(self.sink_results) != len(self.local_results):
            raise ValueError("entry diagnostic omits a planned sink probe")
        for requests, results in (
            (self.local_requests, self.local_results),
            (self.sink_requests, self.sink_results),
        ):
            if len(requests) != len(results) or any(
                result.request != ObjectIdentity.from_record(request.request_id, request)
                for request, result in zip(requests, results, strict=True)
            ):
                raise ValueError("entry diagnostic result belongs to another request")

    @property
    def missing_operands(self) -> tuple[str, ...]:
        reasons = set(self.nontransported_properties)
        if self.horizon_refusal:
            reasons.add("horizon-transport-not-authorized")
        if self.receiver_refusal:
            reasons.add("absolute-temperature-dose-conversion-receiver")
        if any(r.disposition is not LawEvaluationDisposition.SUPPORTED for r in self.sink_results):
            reasons.add("qualified-sink-prediction")
        if any(r.disposition is not LawEvaluationDisposition.SUPPORTED for r in self.local_results):
            reasons.add("supported-local-law-evaluation")
        return tuple(sorted(reasons))


def check_retained_parent(
    parent: ReactorScienceResult, reader: CandidatePayloadReader
) -> ReactorAdmissionEntryDiagnostic:
    """Check the actual retained law, preserving all its identities and cutoffs.

    Expected interface refusals are captured exactly. Unexpected exceptions,
    corrupt payloads and failed custody reads propagate as technical errors;
    they must never become evidence against a controller.
    """
    config, raw, qualification = parent.bound_config, parent.identification, parent.qualification
    if config is None or raw is None or qualification is None or qualification.response_law is None:
        raise ValueError("retained-parent diagnostic requires a qualified local law parent")
    law = qualification.response_law
    publication = raw.candidate_evidence.payload_publication
    if publication is None:
        raise ValueError("retained law lacks its payload publication")
    registration = LawEvaluatorRegistration(
        "law-evaluator.tbs-retained-finite-action",
        "canonical-finite-action-law",
        LawEvaluationKind.FINITE_ACTION_PATH,
        law.representation_kind,
        raw.payload.SCHEMA,
        "finite-action-compatibility-set",
        publication.decoder_schema,
        publication.decoder_version,
        LawEvaluationRequest.SCHEMA,
        LawEvaluationResult.SCHEMA,
        publication.implementation,
        True,
        "resource-envelope.tbs-retained-parent.offline",
        "resource-envelope.tbs-retained-parent.online",
        publication.maximum_decode_bytes,
    )
    owner = LawEvaluationService(
        reader,
        LawEvaluatorRegistry(
            (cast(LawEvaluatorImplementation, FiniteActionLawEvaluator(registration)),)
        ),
    )
    words = {ObjectIdentity.from_record(w.word_id, w): w for w in raw.payload.action_words}
    local_requests, local_results, sink_requests, sink_results = [], [], [], []

    def evaluate(request: LawEvaluationRequest) -> LawEvaluationResult:
        return owner.evaluate(
            config.system, law, qualification, config.method.axis_map, request, publication
        )

    for index, entry in enumerate(raw.payload.entries):
        for view in entry.qualification_view_ids:
            request = LawEvaluationRequest(
                f"reactor-prefix-controller-admission-entry.local.{index}.{view}",
                ObjectIdentity.from_record(law.law_id, law),
                ObjectIdentity.from_record(qualification.result_id, qualification),
                ObjectIdentity.from_record(
                    config.method.axis_map.axis_map_id, config.method.axis_map
                ),
                registration.implementation,
                ObjectIdentity.from_record(publication.receipt_id, publication),
                registration.evaluation_kind,
                LawEvaluationMode.OFFLINE,
                registration.offline_resource_envelope_id,
                None,
                law.system_id,
                law.world_id,
                law.relation.relation_id,
                law.chart_id,
                raw.payload.prepared_denominator_id,
                entry.denominator_member_id,
                entry.candidate_version_id,
                view,
                law.relation.history_quantity_ids,
                law.relation.receiver_quantity_ids,
                raw.payload.horizon,
                entry.support_cell_id,
                words[entry.action_word],
                (),
                (),
                ("response-values",),
            )
            local_requests.append(request)
            local_results.append(evaluate(request))
            sink = replace(
                request,
                request_id=request.request_id + ".sink",
                requested_product_ids=("response-values", "sink-values"),
            )
            sink_requests.append(sink)
            sink_results.append(evaluate(sink))
    if not local_requests:
        raise ValueError("retained payload has no evaluator coordinates")
    target = HorizonSpec(
        horizon_id="reactor-full-batch.horizon",
        clock_id=law.relation.horizon.clock_id,
        duration=Decimal(28800),
        time_unit="s",
    )

    def refusal(request: LawEvaluationRequest, expected: str) -> str:
        try:
            evaluate(request)
        except ValueError as error:
            if str(error) != expected:
                raise
            return str(error)
        return ""

    base = local_requests[0]
    return ReactorAdmissionEntryDiagnostic(
        ObjectIdentity.from_record("tbs-retained-reactor-science", parent),
        law.relation.horizon,
        target,
        tuple(
            sorted(
                {
                    p
                    for axis in config.method.axis_map.bindings
                    for p in axis.nontransported_property_ids
                }
            )
        ),
        tuple(local_requests),
        tuple(local_results),
        tuple(sink_requests),
        tuple(sink_results),
        refusal(
            replace(base, horizon=ObjectIdentity.from_record(target.horizon_id, target)),
            "law evaluation request changes response horizon",
        ),
        refusal(
            replace(
                base,
                receiver_quantity_ids=(
                    "reactor-absolute-temperature",
                    "reactor-conversion",
                    "reactor-dose",
                ),
            ),
            "law evaluation request changes receiver block",
        ),
    )
