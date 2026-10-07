"""Installed task providers for the bounded structural-transport methods."""

from __future__ import annotations

from dataclasses import fields
from decimal import Decimal
from hashlib import sha256
from typing import Protocol, cast, runtime_checkable

from empirical_lawhood.adapters.methods.structural_bootstrap_inputs import StructuralBootstrapInputCensus
from empirical_lawhood.adapters.methods.prospective_structural_recurrence import StructuralRecurrenceFaceApplicability, StructuralRecurrenceFrozenLawTransportForecasts, StructuralRecurrenceMetricDynamicalForecast, StructuralRecurrencePredictiveLevel, ProspectiveStructuralRecurrenceAdjudication, ProspectiveStructuralRecurrenceFaceTerminal, ProspectiveStructuralRecurrenceMethodSpec, ProspectiveStructuralRecurrencePlan, ProspectiveStructuralRecurrenceRosterIssue, ProspectiveStructuralRecurrenceTargetObservation, ProspectiveStructuralRecurrenceTargetTerminal, adjudicate_structural_recurrence_prospective_roster, finalize_structural_recurrence_prospective_face, finalize_structural_recurrence_prospective_target, issue_structural_recurrence_prospective_roster, match_structural_recurrence_prospective_face, observe_structural_recurrence_prospective_face
from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceStageEvidence
from empirical_lawhood.adapters.methods.structural_recurrence import PredictiveMatch, PredictiveStructuralMatcher, StructuralObservation, StructuralPredictionInput
from empirical_lawhood.adapters.methods.interval_property_comparison import IntervalPropertyComparisonMethodSpec, IntervalPropertyComparisonOperands, IntervalPropertyComparisonResult, evaluate_property_comparison as evaluate_interval_property_comparison
from empirical_lawhood.adapters.methods.transformed_property_comparison import TransformedPropertyComparisonMethodSpec, TransformedPropertyComparisonOperands, TransformedPropertyComparisonResult, evaluate_property_comparison as evaluate_transformed_property_comparison
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.laws import ResponseLaw
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ReceiptCheck,
    lineage_parent_sort_key,
)
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.conditional_children import FrozenParentInputBinding
from empirical_lawhood.runtime.law_transport_handoff import AuthenticatedLawTransportHandoff, LAW_TRANSPORT_INPUT_IDS, authenticate_law_transport_handoff
from empirical_lawhood.runtime.response_experiment import ResponseStageTerminal
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .campaign import STRUCTURAL_CAMPAIGN_MEDIA_TYPE, StructuralDevelopmentBridgeConfig, StructuralDevelopmentInputs, StructuralPredictionFreezeReceipt, StructuralRecurrenceTransportConfig, StructuralReporterConfig, StructuralTargetBridgeConfig, StructuralTargetInputs


STRUCTURAL_TRANSPORT_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"


def _read_records(
    context: TaskContext,
    record_types: dict[str, type[CanonicalRecord]],
) -> tuple[CanonicalRecord, ...]:
    records: list[CanonicalRecord] = []
    for port in context.input_ports:
        try:
            record_type = record_types[port.payload_schema]
        except KeyError as error:
            raise ValueError("structural method received an unknown input schema") from error
        payload = port.read(port.size_bytes + 1)
        if len(payload) != port.size_bytes:
            raise ValueError("structural method input size differs from its receipt")
        records.append(decode_canonical_bytes(payload, record_type, maximum_bytes=port.size_bytes))
    return tuple(records)


def _one(records: tuple[CanonicalRecord, ...], kind: type[CanonicalRecord]) -> CanonicalRecord:
    values = tuple(value for value in records if isinstance(value, kind))
    if len(values) != 1:
        raise ValueError(f"structural method requires exactly one {kind.__name__}")
    return values[0]


def _one_or_equal_duplicates(
    records: tuple[CanonicalRecord, ...],
    kind: type[CanonicalRecord],
) -> CanonicalRecord:
    """Accept only byte-identical duplicates from distinct authenticated roles."""

    values = tuple(value for value in records if isinstance(value, kind))
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError(f"structural method requires one exact {kind.__name__} value")
    return values[0]


class IntervalPropertyComparisonTaskRunner:
    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        spec: IntervalPropertyComparisonMethodSpec,
    ) -> None:
        self.manifest = manifest
        self.spec = spec

    def execute(self, context: TaskContext) -> RunnerResult:
        records = _read_records(
            context,
            {
                IntervalPropertyComparisonMethodSpec.SCHEMA: IntervalPropertyComparisonMethodSpec,
                IntervalPropertyComparisonOperands.SCHEMA: IntervalPropertyComparisonOperands,
            },
        )
        spec = _one(records, IntervalPropertyComparisonMethodSpec)
        operands = _one(records, IntervalPropertyComparisonOperands)
        if (
            spec != self.spec
            or context.config.content_sha256 != self.spec.fingerprint()
            or context.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or len(context.output_ports) != 1
            or context.output_ports[0].payload_schema != IntervalPropertyComparisonResult.SCHEMA
        ):
            raise ValueError("property-comparison task differs from its installed contract")
        assert isinstance(operands, IntervalPropertyComparisonOperands)
        result = evaluate_interval_property_comparison(spec=self.spec, operands=operands)
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=result.canonical_bytes(),
                ),
            ),
            checks=(
                ReceiptCheck("property-result-caller-verdict-absent", True, ()),
                ReceiptCheck("property-result-method-derived", True, ()),
            ),
        )


class TransformedPropertyComparisonTaskRunner:
    "Apply typed compatibility transforms for the eight property kinds."

    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        spec: TransformedPropertyComparisonMethodSpec,
    ) -> None:
        self.manifest = manifest
        self.spec = spec

    def execute(self, context: TaskContext) -> RunnerResult:
        records = _read_records(
            context,
            {
                TransformedPropertyComparisonMethodSpec.SCHEMA: TransformedPropertyComparisonMethodSpec,
                TransformedPropertyComparisonOperands.SCHEMA: TransformedPropertyComparisonOperands,
            },
        )
        spec = _one(records, TransformedPropertyComparisonMethodSpec)
        operands = _one(records, TransformedPropertyComparisonOperands)
        if (
            spec != self.spec
            or context.config.content_sha256 != self.spec.fingerprint()
            or context.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or len(context.output_ports) != 1
            or context.output_ports[0].payload_schema != TransformedPropertyComparisonResult.SCHEMA
        ):
            raise ValueError('transformed property comparison task differs from its installed contract')
        assert isinstance(operands, TransformedPropertyComparisonOperands)
        result = evaluate_transformed_property_comparison(spec=self.spec, operands=operands)
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=result.canonical_bytes(),
                ),
            ),
            checks=(
                ReceiptCheck("property-transformed-typed-compatibility-applied", True, ()),
                ReceiptCheck("property-transformed-kind-local-falsifier-applied", True, ()),
            ),
        )


_PROSPECTIVE_STRUCTURAL_RECURRENCE_INPUT_TYPES: dict[str, type[CanonicalRecord]] = {
    StructuralBootstrapInputCensus.SCHEMA: StructuralBootstrapInputCensus,
    ResponseStageTerminal.SCHEMA: ResponseStageTerminal,
    ResponseLaw.SCHEMA: ResponseLaw,
    StructuralRecurrenceFrozenLawTransportForecasts.SCHEMA: StructuralRecurrenceFrozenLawTransportForecasts,
    ProspectiveStructuralRecurrenceMethodSpec.SCHEMA: ProspectiveStructuralRecurrenceMethodSpec,
    ProspectiveStructuralRecurrencePlan.SCHEMA: ProspectiveStructuralRecurrencePlan,
    ProspectiveStructuralRecurrenceRosterIssue.SCHEMA: ProspectiveStructuralRecurrenceRosterIssue,
    StructuralRecurrenceStageEvidence.SCHEMA: StructuralRecurrenceStageEvidence,
    StructuralRecurrenceMetricDynamicalForecast.SCHEMA: StructuralRecurrenceMetricDynamicalForecast,
    IntervalPropertyComparisonResult.SCHEMA: IntervalPropertyComparisonResult,
    StructuralObservation.SCHEMA: StructuralObservation,
    StructuralPredictionInput.SCHEMA: StructuralPredictionInput,
    StructuralRecurrenceTransportConfig.SCHEMA: StructuralRecurrenceTransportConfig,
}


class ProspectiveStructuralRecurrenceTaskRunner:
    """Dispatch issue or target adjudication from the declared output schema."""

    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        spec: ProspectiveStructuralRecurrenceMethodSpec,
        config: CanonicalRecord | None = None,
        handoff: AuthenticatedLawTransportHandoff | None = None,
    ) -> None:
        self.manifest = manifest
        self.spec = spec
        self.config = spec if config is None else config
        self.handoff = handoff

    @staticmethod
    def _categorical_matches(
        issue: object,
        observation: ProspectiveStructuralRecurrenceTargetObservation,
    ) -> tuple[PredictiveMatch, ...]:
        categorical_issue = getattr(issue, "categorical_issue", None)
        if categorical_issue is None or observation.categorical_observation is None:
            return ()
        predictions = (
            categorical_issue.prediction,
            *categorical_issue.comparators.predictions,
        )
        return tuple(
            sorted(
                (
                    PredictiveStructuralMatcher().match(
                        prediction,
                        observation.categorical_observation,
                    )
                    for prediction in predictions
                ),
                key=lambda value: value.match_id,
            )
        )

    def _issue(
        self,
        records: tuple[CanonicalRecord, ...],
    ) -> dict[str, CanonicalRecord]:
        plan = _one_or_equal_duplicates(records, ProspectiveStructuralRecurrencePlan)
        assert isinstance(plan, ProspectiveStructuralRecurrencePlan)
        if plan.method_spec != self.spec:
            raise ValueError('structural recurrence issue plan substitutes its installed method spec')
        evidence = tuple(
            sorted(
                (value for value in records if isinstance(value, StructuralRecurrenceStageEvidence)),
                key=lambda value: value.target_slot_id,
            )
        )
        inputs = tuple(
            sorted(
                (value for value in records if isinstance(value, StructuralPredictionInput)),
                key=lambda value: value.target_slot.target_slot_id,
            )
        )
        frozen = tuple(
            value for value in records if isinstance(value, StructuralRecurrenceFrozenLawTransportForecasts)
        )
        if self.handoff is not None:
            if len(frozen) != 1 or frozen[0] != self.handoff.frozen_forecasts:
                raise ValueError('structural recurrence issue received substituted frozen transport forecasts')
            forecasts = frozen[0].metric_dynamical_forecasts
        else:
            if frozen:
                raise ValueError('standalone structural recurrence issue cannot consume transport forecasts')
            forecasts = tuple(
                sorted(
                    (
                        value
                        for value in records
                        if isinstance(value, StructuralRecurrenceMetricDynamicalForecast)
                    ),
                    key=lambda value: value.forecast_id,
                )
            )
        if self.handoff is not None:
            census = frozen[0].scientific_bootstrap_inputs
            supplied = tuple(value for value in records if isinstance(value, StructuralBootstrapInputCensus))
            if supplied and (len(supplied) != 1 or supplied[0] != census):
                raise ValueError("structural issue substitutes its authenticated numerical census")
        else:
            census = _one(records, StructuralBootstrapInputCensus)
            assert isinstance(census, StructuralBootstrapInputCensus)
        result = issue_structural_recurrence_prospective_roster(plan, evidence, inputs, forecasts, scientific_bootstrap_inputs=census)
        if self.handoff is not None and result != frozen[0].roster_issue:
            raise ValueError('structural recurrence issue differs from the authenticated frozen forecast roster')
        outputs: dict[str, CanonicalRecord] = {"roster-issue": result}
        if self.handoff is not None:
            outputs["prediction-freeze"] = StructuralPredictionFreezeReceipt(
                receipt_id=(
                    "prediction-freeze."
                    + sha256(
                        canonical_json_bytes(
                            (
                                ObjectIdentity.from_record(result.roster_issue_id, result),
                                self.handoff.target_candidate,
                            )
                        )
                    ).hexdigest()[:32]
                ),
                roster_issue=ObjectIdentity.from_record(result.roster_issue_id, result),
                plan=ObjectIdentity.from_record(plan.plan_id, plan),
                method_spec=ObjectIdentity.from_record(self.spec.spec_id, self.spec),
                issued_before_target_contact=True,
                target_contact_count=0,
                target_outcome_access_count=0,
                grants_authority=False,
            )
        return outputs

    def _adjudicate(
        self,
        records: tuple[CanonicalRecord, ...],
    ) -> dict[str, CanonicalRecord]:
        roster = _one(records, ProspectiveStructuralRecurrenceRosterIssue)
        assert isinstance(roster, ProspectiveStructuralRecurrenceRosterIssue)
        if any(
            value.method_spec != ObjectIdentity.from_record(self.spec.spec_id, self.spec)
            for value in roster.face_issues
        ):
            raise ValueError('structural recurrence roster issue substitutes its installed method spec')
        evidence = {
            value.target_slot_id: value for value in records if isinstance(value, StructuralRecurrenceStageEvidence)
        }
        if len(evidence) != sum(isinstance(value, StructuralRecurrenceStageEvidence) for value in records):
            raise ValueError('structural recurrence adjudication repeats target stage evidence')
        property_results = {
            value.face_key: value
            for value in records
            if isinstance(value, IntervalPropertyComparisonResult)
        }
        if len(property_results) != sum(
            isinstance(value, IntervalPropertyComparisonResult) for value in records
        ):
            raise ValueError('structural recurrence adjudication repeats a property face result')
        categorical_observations = {
            value.target_slot_id: value
            for value in records
            if isinstance(value, StructuralObservation)
        }
        if len(categorical_observations) != sum(
            isinstance(value, StructuralObservation) for value in records
        ):
            raise ValueError('structural recurrence adjudication repeats a categorical target observation')

        outputs: dict[str, CanonicalRecord] = {}
        terminals: list[ProspectiveStructuralRecurrenceFaceTerminal] = []
        all_matches: list[PredictiveMatch] = []
        for face in roster.face_specs:
            issue = next(
                value
                for value in roster.face_issues
                if value.face == ObjectIdentity.from_record(face.face_id, face)
            )
            if face.face_applicability is StructuralRecurrenceFaceApplicability.NOT_APPLICABLE:
                terminal = finalize_structural_recurrence_prospective_face(roster, face.face_key)
            elif (target_evidence := evidence.get(face.target_slot_id)) is None:
                terminal = finalize_structural_recurrence_prospective_face(
                    roster,
                    face.face_key,
                    issue=issue,
                )
            else:
                normalized: StructuralObservation | IntervalPropertyComparisonResult | None
                if face.predictive_level is StructuralRecurrencePredictiveLevel.CATEGORICAL:
                    normalized = categorical_observations.get(face.target_slot_id)
                else:
                    normalized = property_results.get(face.face_key)
                if normalized is None:
                    terminal = finalize_structural_recurrence_prospective_face(
                        roster,
                        face.face_key,
                        issue=issue,
                    )
                else:
                    observation = observe_structural_recurrence_prospective_face(
                        issue,
                        target_evidence,
                        normalized,
                    )
                    matches = self._categorical_matches(issue, observation)
                    match_ids = match_structural_recurrence_prospective_face(issue, observation)
                    if match_ids != tuple(
                        ObjectIdentity.from_record(value.match_id, value) for value in matches
                    ):
                        raise ValueError('structural recurrence categorical match reconstruction drifted')
                    terminal = finalize_structural_recurrence_prospective_face(
                        roster,
                        face.face_key,
                        issue=issue,
                        observation=observation,
                        matches=match_ids,
                    )
                    outputs[f"observation.{face.face_id}"] = observation
                    all_matches.extend(matches)
            terminals.append(terminal)
            outputs[f"face-terminal.{face.face_id}"] = terminal

        target_terminals = []
        for target_id in sorted({value.target_slot_id for value in roster.face_specs}):
            target = finalize_structural_recurrence_prospective_target(
                roster,
                target_id,
                tuple(value for value in terminals if value.face_key[0] == target_id),
            )
            target_terminals.append(target)
            outputs[f"target-terminal.{target_id}"] = target
        adjudication = adjudicate_structural_recurrence_prospective_roster(
            roster,
            tuple(terminals),
            tuple(target_terminals),
            tuple(sorted(all_matches, key=lambda value: value.match_id)),
        )
        outputs["adjudication"] = adjudication
        return outputs

    def execute(self, context: TaskContext) -> RunnerResult:
        records = _read_records(context, _PROSPECTIVE_STRUCTURAL_RECURRENCE_INPUT_TYPES)
        specs = tuple(value for value in records if isinstance(value, ProspectiveStructuralRecurrenceMethodSpec))
        if specs:
            spec = _one_or_equal_duplicates(records, ProspectiveStructuralRecurrenceMethodSpec)
        elif isinstance(self.config, StructuralRecurrenceTransportConfig):
            spec = self.config.method_spec
        else:
            raise ValueError('structural recurrence task lacks its installed method spec')
        if spec != self.spec or context.config.content_sha256 != self.config.fingerprint():
            raise ValueError('structural recurrence task differs from its installed method spec')
        output_schemas = {value.payload_schema for value in context.output_ports}
        if output_schemas in (
            {ProspectiveStructuralRecurrenceRosterIssue.SCHEMA},
            {
                ProspectiveStructuralRecurrenceRosterIssue.SCHEMA,
                StructuralPredictionFreezeReceipt.SCHEMA,
            },
        ):
            if context.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
                raise ValueError('structural recurrence issue task crosses the development cutoff')
            if (
                StructuralPredictionFreezeReceipt.SCHEMA in output_schemas
                and self.handoff is None
            ):
                raise ValueError('structural recurrence transport issue lacks its authenticated handoff')
            outputs = self._issue(records)
        elif ProspectiveStructuralRecurrenceAdjudication.SCHEMA in output_schemas:
            if context.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
                raise ValueError('structural recurrence adjudication lacks revealed target inputs')
            outputs = self._adjudicate(records)
        else:
            raise ValueError('structural recurrence task output roster selects no registered operation')
        prefix = f"{context.task_id}."
        ports_by_local_id = {
            (
                value.output_id.removeprefix(prefix)
                if value.output_id.startswith(prefix)
                else value.output_id
            ): value
            for value in context.output_ports
        }
        if len(ports_by_local_id) != len(context.output_ports) or set(outputs) != set(
            ports_by_local_id
        ):
            raise ValueError('structural recurrence task output identities differ from its protocol')
        if any(
            outputs[output_id].SCHEMA != port.payload_schema
            for output_id, port in ports_by_local_id.items()
        ):
            raise ValueError('structural recurrence task output schemas differ from their derived records')
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=outputs[output_id].canonical_bytes(),
                )
                for output_id, port in sorted(ports_by_local_id.items())
            ),
            checks=(
                ReceiptCheck('structural-recurrence-face-roster-complete', True, ()),
                ReceiptCheck('structural-recurrence-no-cross-target-pooling', True, ()),
                ReceiptCheck('structural-recurrence-prospective-cutoff-preserved', True, ()),
            ),
        )


@runtime_checkable
class StructuralTargetSourcePort(Protocol):
    """Effect-bearing target source; called only by the revealed target runner."""

    def load_target_inputs(
        self,
        *,
        config: StructuralTargetBridgeConfig,
        prediction_freeze: StructuralPredictionFreezeReceipt,
    ) -> StructuralTargetInputs: ...

    def jit_signature_projection(self, context: TaskContext) -> object: ...


class StructuralDevelopmentBridgeTaskRunner:
    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        config: StructuralDevelopmentBridgeConfig,
        development_inputs: StructuralDevelopmentInputs,
    ) -> None:
        self.manifest = manifest
        self.config = config
        self.development_inputs = development_inputs

    def execute(self, context: TaskContext) -> RunnerResult:
        records = _read_records(
            context,
            {
                StructuralDevelopmentBridgeConfig.SCHEMA: (StructuralDevelopmentBridgeConfig),
                StructuralDevelopmentInputs.SCHEMA: StructuralDevelopmentInputs,
            },
        )
        config = _one(records, StructuralDevelopmentBridgeConfig)
        inputs = _one(records, StructuralDevelopmentInputs)
        if (
            config != self.config
            or inputs != self.development_inputs
            or context.config.content_sha256 != self.config.fingerprint()
            or context.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        ):
            raise ValueError("development bridge differs from its authenticated contract")
        assert isinstance(inputs, StructuralDevelopmentInputs)
        values: dict[str, CanonicalRecord] = {
            "development": inputs.development_evidence[0],
            "plan": inputs.plan,
            "prediction": inputs.prediction_inputs[0],
            "bootstrap-inputs": inputs.scientific_bootstrap_inputs,
        }
        return _named_outputs(
            context,
            values,
            checks=(
                ReceiptCheck("development-inputs-authenticated", True, ()),
                ReceiptCheck("development-target-outcome-access-zero", True, ()),
            ),
        )


class StructuralTargetBridgeTaskRunner:
    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        config: StructuralTargetBridgeConfig,
        source: StructuralTargetSourcePort,
    ) -> None:
        self.manifest = manifest
        self.config = config
        self.source = source

    def jit_signature_projection(self, context: TaskContext) -> object:
        return self.source.jit_signature_projection(context)

    def execute(self, context: TaskContext) -> RunnerResult:
        records = _read_records(
            context,
            {
                StructuralTargetBridgeConfig.SCHEMA: StructuralTargetBridgeConfig,
                StructuralPredictionFreezeReceipt.SCHEMA: (StructuralPredictionFreezeReceipt),
            },
        )
        config = _one(records, StructuralTargetBridgeConfig)
        freeze = _one(records, StructuralPredictionFreezeReceipt)
        if (
            config != self.config
            or context.config.content_sha256 != self.config.fingerprint()
            or context.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
        ):
            raise ValueError("target bridge crossed its issued reveal contract")
        assert isinstance(freeze, StructuralPredictionFreezeReceipt)
        target = self.source.load_target_inputs(
            config=self.config,
            prediction_freeze=freeze,
        )
        if (
            not isinstance(target, StructuralTargetInputs)
            or target.source_request_id != self.config.source_request_id
            or target.roster_issue != freeze.roster_issue
            or target.evaluation_evidence.target_slot_id != self.config.target_slot_id
            or tuple(sorted(value.comparison_kind.value for value in target.property_operands))
            != self.config.comparison_kinds
        ):
            raise ValueError("target source returned inputs for another issued request")
        values = {
            "categorical": target.categorical_observation,
            "evaluation": target.evaluation_evidence,
            **{
                f"operand.{value.comparison_kind.value.lower()}": value
                for value in target.property_operands
            },
        }
        return _named_outputs(
            context,
            values,
            checks=(
                ReceiptCheck("prediction-freeze-precedes-target-contact", True, ()),
                ReceiptCheck("target-contact-exactly-once", target.source_contact_count == 1, ()),
            ),
        )

    def execute_with_progress(self, context: TaskContext, emitter: object) -> RunnerResult:
        advance = getattr(emitter, "advance", None)
        if not callable(advance):
            raise TypeError("target progress emitter lacks advance")
        advance(Decimal(1))
        return self.execute(context)


class StructuralReporterTaskRunner:
    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        config: StructuralReporterConfig,
    ) -> None:
        self.manifest = manifest
        self.config = config

    def execute(self, context: TaskContext) -> RunnerResult:
        records = _read_records(
            context,
            {
                StructuralReporterConfig.SCHEMA: StructuralReporterConfig,
                ProspectiveStructuralRecurrenceAdjudication.SCHEMA: ProspectiveStructuralRecurrenceAdjudication,
            },
        )
        config = _one(records, StructuralReporterConfig)
        adjudication = _one(records, ProspectiveStructuralRecurrenceAdjudication)
        if (
            config != self.config
            or context.config.content_sha256 != self.config.fingerprint()
            or context.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or len(context.output_ports) != 1
        ):
            raise ValueError("structural reporter differs from its issued contract")
        assert isinstance(adjudication, ProspectiveStructuralRecurrenceAdjudication)
        scientific_context = context.scientific_adjudication_context
        if scientific_context is None:
            raise ValueError("structural reporter lacks the public scientific context")
        record = ScientificAdjudicationRecord(
            adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
            run_id=context.run_id,
            adjudication_task_id=context.task_id,
            execution_plan=scientific_context.execution_plan,
            input_materialization_ids=context.input_materialization_ids,
            output_logical_artifact_ids=tuple(
                sorted(
                    value.logical_artifact_id
                    for value in context.output_ports
                    if value.logical_artifact_id is not None
                )
            ),
            required_receipt_ids=context.dependency_receipt_ids,
            evidence_world_id=scientific_context.evidence_world_id,
            evidence_world_kind=scientific_context.evidence_world_kind,
            relation=scientific_context.relation,
            independent_unit_id=scientific_context.independent_unit_id,
            information_cutoffs=scientific_context.information_cutoffs,
            visibility_ceiling=scientific_context.visibility_ceiling,
            outcome_access=scientific_context.outcome_access,
            evaluability=AdjudicationEvaluability.EVALUABLE,
            scientific_status=(
                ScientificStatus.SUPPORTED
                if adjudication.positive_claim_eligible
                else ScientificStatus.MIXED
            ),
            admission_status=AdmissionStatus.NOT_EVALUATED,
            reason_codes=(adjudication.verdict.value,),
        )
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=record.canonical_bytes(),
                ),
            ),
            checks=(ReceiptCheck('structural-recurrence-adjudication-reported', True, ()),),
        )


def _named_outputs(
    context: TaskContext,
    values: dict[str, CanonicalRecord],
    *,
    checks: tuple[ReceiptCheck, ...],
) -> RunnerResult:
    prefix = f"{context.task_id}."
    ports = {
        (
            value.output_id.removeprefix(prefix)
            if value.output_id.startswith(prefix)
            else value.output_id
        ): value
        for value in context.output_ports
    }
    if len(ports) != len(context.output_ports) or set(ports) != set(values):
        raise ValueError("structural campaign output roster differs from its protocol")
    if any(values[name].SCHEMA != port.payload_schema for name, port in ports.items()):
        raise ValueError("structural campaign output schema differs from its record")
    return RunnerResult(
        outputs=tuple(
            TaskOutputPayload(output_id=port.output_id, payload=values[name].canonical_bytes())
            for name, port in sorted(ports.items())
        ),
        checks=checks,
    )


def _record_id(record: CanonicalRecord) -> str:
    for name in (
        "config_id",
        "inputs_id",
        "terminal_id",
        "law_id",
        "spec_id",
        "plan_id",
        "forecast_id",
        "record_id",
    ):
        value = getattr(record, name, None)
        if isinstance(value, str):
            return value
    raise ValueError("structural external record lacks a stable identity field")


def _external_payload(specification: object, record: CanonicalRecord) -> ExternalInputPayload:
    logical_artifact_id = getattr(specification, "logical_artifact_id")
    expected_schema = getattr(specification, "expected_payload_schema")
    expected_media_type = getattr(specification, "expected_media_type")
    expected_content_sha256 = getattr(specification, "expected_content_sha256")
    visibility = getattr(specification, "expected_visibility_ceiling") or (
        VisibilityCeiling.PROSPECTIVE
    )
    access = getattr(specification, "expected_outcome_access") or OutcomeAccess.OUTCOME_BLIND
    if (
        record.SCHEMA != expected_schema
        or expected_media_type not in {None, STRUCTURAL_CAMPAIGN_MEDIA_TYPE}
        or (expected_content_sha256 is not None and expected_content_sha256 != record.fingerprint())
    ):
        raise ValueError("structural external record differs from its compiled input spec")
    parents = [
        ArtifactLineageParent(
            identity=ObjectIdentity.from_record(_record_id(record), record),
            visibility_ceiling=visibility,
            outcome_access=access,
        )
    ]
    identity_scope_sha256 = getattr(specification, "identity_scope_sha256")
    if identity_scope_sha256 is not None:
        parents.append(
            ArtifactLineageParent(
                identity=ObjectIdentity(
                    object_id=f"external-input-scope.{getattr(specification, 'input_id')}",
                    object_schema='empirical-lawhood/runtime/external-input-scope',
                    object_version="1.0.0",
                    object_fingerprint=identity_scope_sha256,
                ),
                visibility_ceiling=visibility,
                outcome_access=access,
            )
        )
    lineage = tuple(sorted(parents, key=lineage_parent_sort_key))
    return ExternalInputPayload.from_bytes(
        logical_artifact_id=logical_artifact_id,
        payload_schema=record.SCHEMA,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type=STRUCTURAL_CAMPAIGN_MEDIA_TYPE,
        payload=record.canonical_bytes(),
        visibility_ceiling=visibility,
        outcome_access=access,
        parent_visibility_ceilings=tuple(value.visibility_ceiling for value in lineage),
        lineage_parents=lineage,
        logical_content_sha256=expected_content_sha256,
    )


class _StructuralCampaignSingleProvider(CampaignRuntimeProvider):
    issued_source_schema_ids: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: CanonicalRecord,
        runner: TaskRunner,
        external_records: tuple[CanonicalRecord, ...],
        output_types: tuple[type[CanonicalRecord], ...],
        adjudication_output_id: str | None = None,
    ) -> None:
        if registry.resolve(manifest.capability_key, manifest.capability_version) != manifest:
            raise ValueError("structural campaign manifest differs from its registry")
        schemas = tuple(value.SCHEMA for value in (config, *external_records))
        if len(set(schemas)) != len(schemas):
            raise ValueError("structural provider construction records overlap schemas")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.manifest = manifest
        self.config = config
        self._runner = runner
        self._records = {value.SCHEMA: value for value in (config, *external_records)}
        self._output_types: tuple[type[CanonicalRecord], ...] = output_types
        self._adjudication_output_id = adjudication_output_id

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("structural campaign provider registry/source records differ")
        return (self._runner,)

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("structural campaign provider plan/source records differ")
        specifications = {
            value.logical_artifact_id: value
            for task in plan.tasks
            if task.capability.capability_key == self.manifest.capability_key
            and task.capability.capability_version == self.manifest.capability_version
            for value in task.external_inputs
        }
        payloads = []
        for specification in specifications.values():
            expected_schema = specification.expected_payload_schema
            if expected_schema is None:
                raise ValueError("structural provider input lacks an expected payload schema")
            record = self._records.get(expected_schema)
            if record is None:
                raise ValueError("structural provider lacks one compiled external input")
            payloads.append(_external_payload(specification, record))
        return tuple(sorted(payloads, key=lambda value: value.logical_artifact_id))

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("structural campaign semantic registry differs")
        del execution_plan
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest,
                payload_schema=record_type.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(
                    sorted(
                        value.name
                        for value in fields(record_type)  # type: ignore[arg-type]
                    )
                ),
            )
            for record_type in sorted(self._output_types, key=lambda value: value.SCHEMA)
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract | None:
        if registry != self.registry:
            raise ValueError("structural campaign adjudication registry differs")
        del execution_plan
        if self._adjudication_output_id is None:
            return None
        return ScientificAdjudicationOutputContract(
            capability_key=self.manifest.capability_key,
            capability_version=self.manifest.capability_version,
            output_id=self._adjudication_output_id,
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            maximum_bytes=128 * 1024,
        )


class StructuralRecurrenceTransportProvider(CampaignRuntimeProvider):
    issued_source_schema_ids = (FrozenParentInputBinding.SCHEMA,)

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: StructuralRecurrenceTransportConfig,
    ) -> None:
        if registry.resolve(manifest.capability_key, manifest.capability_version) != manifest:
            raise ValueError('structural structural recurrence transport manifest differs from its registry')
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.manifest = manifest
        self.config = config

    def _handoff(
        self,
        source_records: tuple[CanonicalRecord, ...],
    ) -> AuthenticatedLawTransportHandoff:
        if len(source_records) != len(LAW_TRANSPORT_INPUT_IDS) or not all(
            isinstance(value, FrozenParentInputBinding) for value in source_records
        ):
            raise ValueError('structural structural recurrence provider lacks five parent-input bindings')
        bindings = cast(tuple[FrozenParentInputBinding, ...], tuple(source_records))
        first = bindings[0]
        assert isinstance(first, FrozenParentInputBinding)
        return authenticate_law_transport_handoff(
            target_candidate=first.candidate,
            scientific_graph_sha256=first.scientific_graph_sha256,
            bindings=bindings,
            forecast_method_spec_type=ProspectiveStructuralRecurrenceMethodSpec,
            forecast_method_config_type=ProspectiveStructuralRecurrencePlan,
            frozen_forecasts_type=StructuralRecurrenceFrozenLawTransportForecasts,
        )

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry:
            raise ValueError('structural structural recurrence provider registry differs')
        handoff = self._handoff(source_records)
        if (
            not isinstance(handoff.forecast_method_config, ProspectiveStructuralRecurrencePlan)
            or not isinstance(handoff.forecast_method_spec, ProspectiveStructuralRecurrenceMethodSpec)
            or handoff.forecast_method_spec != self.config.method_spec
            or ObjectIdentity.from_record(
                handoff.forecast_method_config.plan_id,
                handoff.forecast_method_config,
            )
            != self.config.plan
        ):
            raise ValueError('structural structural recurrence config differs from its authenticated handoff')
        return (
            ProspectiveStructuralRecurrenceTaskRunner(
                manifest=self.manifest,
                spec=self.config.method_spec,
                config=self.config,
                handoff=handoff,
            ),
        )

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256:
            raise ValueError('structural structural recurrence provider plan differs')
        handoff = self._handoff(source_records)
        bindings = {value.external_input_id: value for value in handoff.bindings}
        specifications = {
            value.logical_artifact_id: value
            for task in plan.tasks
            if task.capability.capability_key == self.manifest.capability_key
            and task.capability.capability_version == self.manifest.capability_version
            for value in task.external_inputs
        }
        payloads = []
        for specification in specifications.values():
            if specification.expected_payload_schema == self.config.SCHEMA:
                record: CanonicalRecord = self.config
            else:
                input_id = specification.input_id
                if input_id is None:
                    raise ValueError('structural structural recurrence input lacks its parent input ID')
                binding = bindings.get(input_id)
                if binding is None:
                    raise ValueError('structural structural recurrence external input lacks a parent binding')
                record_types: dict[str, type[CanonicalRecord]] = {
                    ResponseStageTerminal.SCHEMA: ResponseStageTerminal,
                    ResponseLaw.SCHEMA: ResponseLaw,
                    ProspectiveStructuralRecurrenceMethodSpec.SCHEMA: ProspectiveStructuralRecurrenceMethodSpec,
                    ProspectiveStructuralRecurrencePlan.SCHEMA: ProspectiveStructuralRecurrencePlan,
                    StructuralRecurrenceFrozenLawTransportForecasts.SCHEMA: (StructuralRecurrenceFrozenLawTransportForecasts),
                }
                expected_schema = specification.expected_payload_schema
                if expected_schema is None:
                    raise ValueError('structural structural recurrence input lacks an expected payload schema')
                record_type = record_types.get(expected_schema)
                if record_type is None:
                    raise ValueError('structural structural recurrence input names another parent schema')
                record = binding.decode_parent(record_type, maximum_bytes=8_000_000)
            payloads.append(_external_payload(specification, record))
        return tuple(sorted(payloads, key=lambda value: value.logical_artifact_id))

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError('structural structural recurrence semantic registry differs')
        del execution_plan
        output_types: tuple[type[CanonicalRecord], ...] = (
            StructuralPredictionFreezeReceipt,
            ProspectiveStructuralRecurrenceRosterIssue,
            ProspectiveStructuralRecurrenceTargetObservation,
            ProspectiveStructuralRecurrenceFaceTerminal,
            ProspectiveStructuralRecurrenceTargetTerminal,
            ProspectiveStructuralRecurrenceAdjudication,
        )
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest,
                payload_schema=record_type.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(
                    sorted(
                        value.name
                        for value in fields(record_type)  # type: ignore[arg-type]
                    )
                ),
            )
            for record_type in sorted(output_types, key=lambda value: value.SCHEMA)
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError('structural structural recurrence adjudication registry differs')
        del execution_plan
        return None


class _SingleManifestProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: CanonicalRecord,
        runner: TaskRunner,
    ) -> None:
        installed = registry.resolve(manifest.capability_key, manifest.capability_version)
        if installed != manifest:
            raise ValueError("structural method manifest differs from the installed registry")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.manifest = manifest
        self.config = config
        self._runner = runner

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("structural provider registry/source records differ")
        return (self._runner,)

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("structural provider plan/source records differ")
        specs = {
            task.capability.config.artifact_id: spec
            for task in plan.tasks
            if task.capability.capability_key == self.manifest.capability_key
            and task.capability.capability_version == self.manifest.capability_version
            for spec in task.external_inputs
            if spec.logical_artifact_id == task.capability.config.artifact_id
        }
        requested = {
            spec.logical_artifact_id
            for task in plan.tasks
            if task.capability.capability_key == self.manifest.capability_key
            and task.capability.capability_version == self.manifest.capability_version
            for spec in task.external_inputs
        }
        if requested != set(specs):
            raise ValueError("structural provider accepts only its issued method config externally")
        identity = ObjectIdentity.from_record(getattr(self.config, "spec_id"), self.config)
        return tuple(
            ExternalInputPayload.from_bytes(
                logical_artifact_id=artifact_id,
                payload_schema=self.config.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type=STRUCTURAL_TRANSPORT_MEDIA_TYPE,
                payload=self.config.canonical_bytes(),
                visibility_ceiling=spec.expected_visibility_ceiling
                or VisibilityCeiling.PROSPECTIVE,
                outcome_access=spec.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND,
                parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                lineage_parents=(
                    ArtifactLineageParent(
                        identity=identity,
                        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                        outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    ),
                ),
                logical_content_sha256=spec.expected_content_sha256,
            )
            for artifact_id, spec in sorted(specs.items())
        )

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("structural provider semantic registry differs")
        del execution_plan
        output_types = {
            value.SCHEMA: value
            for value in (
                IntervalPropertyComparisonResult,
                TransformedPropertyComparisonResult,
                ProspectiveStructuralRecurrenceRosterIssue,
                ProspectiveStructuralRecurrenceTargetObservation,
                ProspectiveStructuralRecurrenceFaceTerminal,
                ProspectiveStructuralRecurrenceTargetTerminal,
                ProspectiveStructuralRecurrenceAdjudication,
            )
            if value.SCHEMA in self.manifest.output_schema_ids
        }
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest,
                payload_schema=schema,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(value.name for value in fields(record_type))),
            )
            for schema, record_type in sorted(output_types.items())
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("structural provider adjudication registry differs")
        del execution_plan
        return None


def interval_property_comparison_provider(
    *,
    registry: CapabilityRegistry,
    manifest: CapabilityManifest,
    spec: IntervalPropertyComparisonMethodSpec,
) -> CampaignRuntimeProvider:
    return _SingleManifestProvider(
        registry=registry,
        manifest=manifest,
        config=spec,
        runner=IntervalPropertyComparisonTaskRunner(manifest=manifest, spec=spec),
    )


def transformed_property_comparison_provider(
    *,
    registry: CapabilityRegistry,
    manifest: CapabilityManifest,
    spec: TransformedPropertyComparisonMethodSpec,
) -> CampaignRuntimeProvider:
    return _SingleManifestProvider(
        registry=registry,
        manifest=manifest,
        config=spec,
        runner=TransformedPropertyComparisonTaskRunner(manifest=manifest, spec=spec),
    )


def structural_recurrence_prospective_provider(
    *,
    registry: CapabilityRegistry,
    manifest: CapabilityManifest,
    spec: ProspectiveStructuralRecurrenceMethodSpec,
) -> CampaignRuntimeProvider:
    return _SingleManifestProvider(
        registry=registry,
        manifest=manifest,
        config=spec,
        runner=ProspectiveStructuralRecurrenceTaskRunner(manifest=manifest, spec=spec),
    )


def structural_development_bridge_provider(
    *,
    registry: CapabilityRegistry,
    manifest: CapabilityManifest,
    config: StructuralDevelopmentBridgeConfig,
    development_inputs: StructuralDevelopmentInputs,
) -> CampaignRuntimeProvider:
    return _StructuralCampaignSingleProvider(
        registry=registry,
        manifest=manifest,
        config=config,
        runner=StructuralDevelopmentBridgeTaskRunner(
            manifest=manifest,
            config=config,
            development_inputs=development_inputs,
        ),
        external_records=(development_inputs,),
        output_types=(ProspectiveStructuralRecurrencePlan, StructuralRecurrenceStageEvidence, StructuralPredictionInput, StructuralBootstrapInputCensus),
    )


def structural_target_bridge_provider(
    *,
    registry: CapabilityRegistry,
    manifest: CapabilityManifest,
    config: StructuralTargetBridgeConfig,
    source: StructuralTargetSourcePort,
) -> CampaignRuntimeProvider:
    return _StructuralCampaignSingleProvider(
        registry=registry,
        manifest=manifest,
        config=config,
        runner=StructuralTargetBridgeTaskRunner(
            manifest=manifest,
            config=config,
            source=source,
        ),
        external_records=(),
        output_types=(StructuralRecurrenceStageEvidence, StructuralObservation, IntervalPropertyComparisonOperands),
    )


def structural_reporter_provider(
    *,
    registry: CapabilityRegistry,
    manifest: CapabilityManifest,
    config: StructuralReporterConfig,
) -> CampaignRuntimeProvider:
    return _StructuralCampaignSingleProvider(
        registry=registry,
        manifest=manifest,
        config=config,
        runner=StructuralReporterTaskRunner(manifest=manifest, config=config),
        external_records=(),
        output_types=(ScientificAdjudicationRecord,),
        adjudication_output_id="report.report",
    )


def structural_structural_recurrence_transport_provider(
    *,
    registry: CapabilityRegistry,
    manifest: CapabilityManifest,
    config: StructuralRecurrenceTransportConfig,
) -> CampaignRuntimeProvider:
    return StructuralRecurrenceTransportProvider(
        registry=registry,
        manifest=manifest,
        config=config,
    )


__all__ = [
    'ProspectiveStructuralRecurrenceTaskRunner',
    'IntervalPropertyComparisonTaskRunner',
    'TransformedPropertyComparisonTaskRunner',
    "STRUCTURAL_TRANSPORT_MEDIA_TYPE",
    'StructuralDevelopmentBridgeTaskRunner',
    'StructuralRecurrenceTransportProvider',
    'StructuralReporterTaskRunner',
    'StructuralTargetBridgeTaskRunner',
    'StructuralTargetSourcePort',
    'interval_property_comparison_provider',
    'transformed_property_comparison_provider',
    'structural_recurrence_prospective_provider',
    'structural_development_bridge_provider',
    'structural_structural_recurrence_transport_provider',
    'structural_reporter_provider',
    'structural_target_bridge_provider',
]
